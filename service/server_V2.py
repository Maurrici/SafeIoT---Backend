import joblib
import numpy as np
import pandas as pd
import json
import time
import ssl
from collections import deque
from scipy.stats import skew, kurtosis
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
import paho.mqtt.client as mqtt
import warnings

# Ignorar avisos de versão
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. CONFIGURAÇÕES HIVEMQ CLOUD
# ==============================================================================
MQTT_BROKER = "4f97c41ebc1e4119a1b005fe735cd462.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USER = "detector_queda"
MQTT_PASS = "trabalhoIOT1"
TOPIC_INPUT = "data"         # Tópico onde o ESP32 publica
TOPIC_OUTPUT = "quedas"      # Tópico para avisar o Dashboard

# Configurações de Buffer
FREQUENCY = 25.0             # Hz
WINDOW_SECONDS = 3
WINDOW_SIZE = int(FREQUENCY * WINDOW_SECONDS)  # ~75 amostras
BUFFER_MAX = WINDOW_SIZE * 2              
COOLDOWN_SEC = 5             # Tempo para não disparar alertas repetidos

# ==============================================================================
# 2. CLASSE GMM (Necessária para o Joblib reconhecer o objeto)
# ==============================================================================
class GMMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components_0=1, n_components_1=1, 
                 covariance_type_0='full', covariance_type_1='full', threshold=0.0):
        self.n_components_0 = n_components_0
        self.n_components_1 = n_components_1
        self.covariance_type_0 = covariance_type_0
        self.covariance_type_1 = covariance_type_1
        self.threshold = threshold
        self.gmm_0 = None
        self.gmm_1 = None
        self.priors = None

    def fit(self, X, y):
        # ... (código de fit omitido pois só precisamos carregar)
        return self

    def predict_score(self, X):
        X = np.array(X)
        log_prob_0 = self.gmm_0.score_samples(X) + np.log(self.priors[0])
        log_prob_1 = self.gmm_1.score_samples(X) + np.log(self.priors[1])
        return log_prob_1 - log_prob_0

    def predict(self, X):
        diff = self.predict_score(X)
        return (diff > self.threshold).astype(int)

# ==============================================================================
# 3. EXTRAÇÃO DE FEATURES (Versão Otimizada)
# ==============================================================================
def extract_features(window):
    # 1. Preparação Básica
    window = np.array(window)

    if window.size == 0 or window.ndim < 2 or window.shape[1] < 6:
        return None

    acc = window[:, :3]
    gyro = window[:, 3:]

    # Magnitudes (Norma Vetorial)
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    fs = FREQUENCY 

    # Jerk (Derivada)
    if len(acc_mag) > 1:
        jerk = np.diff(acc_mag) * fs
    else:
        jerk = np.array([0.0])

    # --- FUNÇÕES SEGURAS ---
    def safe_skew(x):
        try:
            val = skew(x)
            return 0.0 if np.isnan(val) else float(val)
        except: return 0.0

    def safe_kurtosis(x):
        try:
            val = kurtosis(x)
            return 0.0 if np.isnan(val) else float(val)
        except: return 0.0

    feats = {}

    # 2. ACELERÔMETRO
    acc_mean = np.mean(acc_mag)
    acc_max = np.max(acc_mag)
    acc_min = np.min(acc_mag)
    
    feats['acc_mag_mean']     = float(acc_mean)
    feats['acc_mag_std']      = float(np.std(acc_mag))
    feats['acc_mag_min']      = float(acc_min)
    feats['acc_mag_max']      = float(acc_max)
    feats['acc_mag_range']    = float(acc_max - acc_min)
    feats['acc_mag_p95']      = float(np.percentile(acc_mag, 95))
    feats['acc_mag_rms']      = float(np.sqrt(np.mean(acc_mag**2)))
    
    # IMPORTANTE: No dicionário a ordem não importa tanto, 
    # mas note que SMA é calculado aqui
    feats['acc_mag_sma']      = float(np.sum(np.abs(acc_mag - acc_mean)) / len(acc_mag))
    feats['acc_mag_skew']     = safe_skew(acc_mag)
    feats['acc_mag_kurtosis'] = safe_kurtosis(acc_mag)

    # 3. GIROSCÓPIO
    gyro_mean = np.mean(gyro_mag)
    gyro_max = np.max(gyro_mag)
    gyro_min = np.min(gyro_mag)

    feats['gyro_mag_mean']  = float(gyro_mean)
    feats['gyro_mag_std']   = float(np.std(gyro_mag))
    feats['gyro_mag_max']   = float(gyro_max)
    feats['gyro_mag_range'] = float(gyro_max - gyro_min)
    feats['gyro_mag_p95']   = float(np.percentile(gyro_mag, 95))
    feats['gyro_mag_rms']   = float(np.sqrt(np.mean(gyro_mag**2)))
    feats['gyro_mag_sma']   = float(np.sum(np.abs(gyro_mag - gyro_mean)) / len(gyro_mag))
    
    # 4. JERK & IMPACTO
    if len(jerk) > 0:
        feats['jerk_mean'] = float(np.mean(np.abs(jerk)))
        feats['jerk_std']  = float(np.std(jerk))
        feats['jerk_max']  = float(np.max(np.abs(jerk)))
    else:
        feats['jerk_mean'] = 0.0
        feats['jerk_std']  = 0.0
        feats['jerk_max']  = 0.0

    # Impact Ratio
    n_post = int(len(acc_mag) * 0.3)
    if n_post > 0:
        post_mean = np.mean(acc_mag[-n_post:])
    else:
        post_mean = 9.8

    feats['impact_ratio'] = float(acc_max / (post_mean + 1e-6))

    return feats

# ==============================================================================
# 4. SISTEMA MQTT WORKER
# ==============================================================================

class MQTTFallDetector:
    def __init__(self, model_file, scaler_file):
        print(f"[INIT] Carregando Sistema...")
        
        try:
            # 1. Carrega Modelo e Scaler de arquivos separados
            self.model = joblib.load(model_file)
            print(f" -> Modelo carregado: {model_file}")
            
            self.scaler = joblib.load(scaler_file)
            print(f" -> Scaler carregado: {scaler_file}")
            
            # 2. Define a ordem exata das colunas
            self.features_esperadas = [
                "acc_mag_mean", "acc_mag_std", "acc_mag_min", "acc_mag_max",
                "acc_mag_range", "acc_mag_p95", "acc_mag_rms",
                "acc_mag_skew",     # <-- Skew antes
                "acc_mag_kurtosis", # <-- Kurtosis antes
                "acc_mag_sma",      # <-- SMA depois
                
                "gyro_mag_mean", "gyro_mag_std", "gyro_mag_max", "gyro_mag_range",
                "gyro_mag_p95", "gyro_mag_rms", "gyro_mag_sma",
                "jerk_mean", "jerk_std", "jerk_max", "impact_ratio"
            ]
            
            print("[INIT] Sucesso!")
            
        except Exception as e:
            print(f"[ERRO CRÍTICO] Falha ao carregar arquivos .pkl: {e}")
            print(f"Verifique se '{model_file}' e '{scaler_file}' estão na mesma pasta.")
            exit(1)
        
        self.buffer = deque(maxlen=BUFFER_MAX)
        self.last_fall_time = 0
        self.msg_count = 0 # <-- Contador para controlar prints

        # Configuração MQTT
        self.client = mqtt.Client()
        self.client.username_pw_set(MQTT_USER, MQTT_PASS)
        
        # SSL Context para HiveMQ Cloud
        context = ssl.create_default_context()
        self.client.tls_set_context(context)
        
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        print(f"📡 [CONN] Conectando ao HiveMQ ({MQTT_BROKER})...")
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        except Exception as e:
            print(f"[ERRO] Falha de conexão: {e}")
            exit(1)

    def start(self):
        print("[LOOP] Servidor Rodando! Pressione Ctrl+C para parar.")
        self.client.loop_forever()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[CONN] Conectado! Inscrito em: '{TOPIC_INPUT}'")
            client.subscribe(TOPIC_INPUT)
        else:
            print(f"[CONN] Erro de conexão. Código: {rc}")

    def on_message(self, client, userdata, msg):
        try:
            raw_str = msg.payload.decode()
            payload = json.loads(raw_str)
            data_chunk = []
            
            # [DEBUG] Ponto de vida a cada mensagem recebida
            print(".", end="", flush=True)
            self.msg_count += 1

            # --- Lógica de Parsing ---
            if isinstance(payload, dict) and 'ax' in payload:
                try:
                    row = [
                        float(payload['ax']), float(payload['ay']), float(payload['az']),
                        float(payload['gx']), float(payload['gy']), float(payload['gz'])
                    ]
                    data_chunk = [row]
                except KeyError:
                    pass

            elif isinstance(payload, dict) and 'data' in payload:
                data_chunk = payload['data']
            
            elif isinstance(payload, list):
                data_chunk = payload

            if data_chunk:
                self.ingest_data(data_chunk)

        except json.JSONDecodeError:
            print("x", end="", flush=True)
        except Exception as e:
            print(f"\n[ERRO MSG] {e}")

    def ingest_data(self, data_chunk):
        current_server_time = time.time()
        step = 1.0 / FREQUENCY
        
        if len(data_chunk) > 0 and len(data_chunk[0]) != 6:
            return

        for i, row in enumerate(data_chunk):
            estimated_ts = current_server_time - ((len(data_chunk) - 1 - i) * step)
            full_row = [estimated_ts] + row 
            self.buffer.append(full_row)
        
        if len(self.buffer) >= WINDOW_SIZE:
            self._process_window()

    def _process_window(self):
        window_list = list(self.buffer)[-WINDOW_SIZE:]
        window_array = np.array(window_list)

        timestamps = window_array[:, 0]
        sensor_data = window_array[:, 1:7].astype(float)

        try:
            # 1. Extrair Features
            feats = extract_features(sensor_data)
            
            if feats is None: return

            # 2. Criar DataFrame 
            df_feats = pd.DataFrame([feats])
            
            # --- CORREÇÃO: Forçar a ordem das colunas ---
            df_feats = df_feats[self.features_esperadas]
            
            # 3. Escalar
            X_scaled = self.scaler.transform(df_feats)
            
            # 4. Predizer (Usando Score para debug melhor)
            raw_score = self.model.predict_score(X_scaled)[0]
            prediction = 1 if raw_score > self.model.threshold else 0
            
            # [DEBUG] ANALISE PERIÓDICA (A cada ~25 mensagens para não poluir)
            if self.msg_count % 25 == 0:
                status = "QUEDA" if prediction == 1 else "Normal"
                # Acc deve ser ~9.8. Se o Score > 0.0 é queda.
                print(f"\n[ANALISE] Acc: {feats['acc_mag_mean']:.2f}m/s^2 | Score: {raw_score:.3f} | Status: {status}")

            if prediction == 1:
                now = time.time()
                if (now - self.last_fall_time) > COOLDOWN_SEC:
                    print(f"\n🚨🚨🚨 QUEDA DETECTADA! (Score: {raw_score:.2f}) às {time.ctime(now)}")
                    self.last_fall_time = now
                    self._publish_alert(timestamps[-1])
            else:
                pass

        except Exception as e:
            print(f"\n[ERRO INF] {e}")

    def _publish_alert(self, ts_evento):
        payload = json.dumps({
            "event": "FALL_DETECTED",
            "timestamp": ts_evento,
            "confidence": "high"
        })
        try:
            self.client.publish(TOPIC_OUTPUT, payload, qos=1)
            # CHAMADA PARA API
            print(f" -> Alerta enviado para MQTT: {TOPIC_OUTPUT}")
        except Exception as e:
            print(f" -> Falha no envio MQTT: {e}")

if __name__ == '__main__':
    # Certifique-se que o nome do arquivo está correto
    MODEL_FILE = 'gmm_classifier_V1.pkl'
    SCALER_FILE = 'scaler_padrao.pkl'
    
    detector = MQTTFallDetector(MODEL_FILE, SCALER_FILE)
    detector.start()