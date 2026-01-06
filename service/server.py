import joblib
import numpy as np
import pandas as pd
import json
import time
import ssl  # <--- Necessário para HiveMQ (Porta 8883)
from collections import deque
from scipy.stats import skew, kurtosis
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
import paho.mqtt.client as mqtt

# ==============================================================================
# 1. CONFIGURAÇÕES HIVEMQ CLOUD
# ==============================================================================
MQTT_BROKER = "4f97c41ebc1e4119a1b005fe735cd462.s1.eu.hivemq.cloud"
MQTT_PORT = 8883             # Porta segura
MQTT_USER = "detector_queda"
MQTT_PASS = "trabalhoIOT1"
TOPIC_INPUT = "data"         # Ler dados daqui
TOPIC_OUTPUT = "quedas"      # Escrever alertas aqui

# Configurações de Buffer
FREQUENCY = 20     
WINDOW_SECONDS = 3
WINDOW_SIZE = FREQUENCY * WINDOW_SECONDS  # 75 amostras
BUFFER_MAX = WINDOW_SIZE * 2              
COOLDOWN_SEC = 5   

# ==============================================================================
# 2. CLASSE GMM (Obrigatória para o Joblib)
# ==============================================================================
class GMMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components_0=1, n_components_1=1, covariance_type='full', threshold=0.0):
        self.n_components_0 = n_components_0
        self.n_components_1 = n_components_1
        self.covariance_type = covariance_type
        self.threshold = threshold
        self.gmm_0 = None
        self.gmm_1 = None
        self.priors = None
    def fit(self, X, y): pass 
    def predict_score(self, X):
        log_prob_0 = self.gmm_0.score_samples(X) + np.log(self.priors[0])
        log_prob_1 = self.gmm_1.score_samples(X) + np.log(self.priors[1])
        return log_prob_1 - log_prob_0
    def predict(self, X):
        diff = self.predict_score(X)
        return (diff > self.threshold).astype(int)

# ==============================================================================
# 3. EXTRAÇÃO DE FEATURES
# ==============================================================================
def extract_features(window_array):
    acc = window_array[:, :3]
    gyro = window_array[:, 3:]
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    fs = 25 
    jerk = np.diff(acc_mag) * fs

    def safe_skew(x):
        try: return 0.0 if np.isnan(s := skew(x)) else s
        except: return 0.0
    def safe_kurtosis(x):
        try: return 0.0 if np.isnan(k := kurtosis(x)) else k
        except: return 0.0
    def stats(x, prefix):
        return {
            f"{prefix}_mean": float(np.mean(x)),
            f"{prefix}_std":  float(np.std(x)),
            f"{prefix}_min":  float(np.min(x)),
            f"{prefix}_max":  float(np.max(x)),
            f"{prefix}_range": float(np.max(x) - np.min(x)),
            f"{prefix}_p95":  float(np.percentile(x, 95)),
            f"{prefix}_energy": float(np.sum(x**2) / len(x)),
            f"{prefix}_rms": float(np.sqrt(np.mean(x**2))),
            f"{prefix}_skew": safe_skew(x),
            f"{prefix}_kurtosis": safe_kurtosis(x),
        }
    feats = {}
    feats.update(stats(acc_mag, "acc_mag"))
    feats.update(stats(gyro_mag, "gyro_mag"))
    feats.update({
        "jerk_mean": float(np.mean(np.abs(jerk))) if len(jerk) > 0 else 0.0,
        "jerk_std" : float(np.std(jerk)) if len(jerk) > 0 else 0.0,
        "jerk_max" : float(np.max(np.abs(jerk))) if len(jerk) > 0 else 0.0,
    })
    return feats

# ==============================================================================
# 4. SISTEMA MQTT WORKER
# ==============================================================================

class MQTTFallDetector:
    def __init__(self, model_path):
        print(f"[INIT] Carregando IA ({model_path})...")
        try:
            pacote = joblib.load(model_path)
            self.model = pacote['modelo']
            self.scaler = pacote['scaler']
            self.features_esperadas = pacote['features']
            print("[INIT] Modelo carregado com sucesso.")
        except Exception as e:
            print(f"[ERRO] Falha ao carregar modelo: {e}")
            exit(1)
        
        self.buffer = deque(maxlen=BUFFER_MAX)
        self.last_fall_time = 0

        self.client = mqtt.Client()
        self.client.username_pw_set(MQTT_USER, MQTT_PASS)
        self.client.tls_set(cert_reqs=ssl.CERT_NONE) 
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        print(f"📡 [CONN] Tentando conectar ao HiveMQ ({MQTT_BROKER})...")
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        except Exception as e:
            print(f"[ERRO] Falha na conexão inicial: {e}")
            exit(1)

    def start(self):
        print("[LOOP] Servidor Rodando! Aguardando dados...")
        self.client.loop_forever()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"[CONN] Conectado! Inscrevendo no tópico: '{TOPIC_INPUT}'")
            client.subscribe(TOPIC_INPUT)
        else:
            print(f"[CONN] Falha ao conectar. Código de retorno: {rc}")

    def on_message(self, client, userdata, msg):
        # DEBUG 1: Saber se a mensagem chegou fisicamente
        # print(f"[MSG] Recebido no tópico '{msg.topic}' | Tamanho: {len(msg.payload)} bytes")
        
        try:
            raw_str = msg.payload.decode()
            payload = json.loads(raw_str)
            
            data_chunk = []

            # --- CORREÇÃO AQUI ---
            # CASO 1: Formato Dicionário Único (O que está chegando agora)
            if isinstance(payload, dict) and 'ax' in payload and 'gx' in payload:
                # O resto do código espera uma lista de listas: [[ax, ay, az, gx, gy, gz]]
                # Precisamos garantir a ordem exata das colunas!
                try:
                    row = [
                        float(payload['ax']),
                        float(payload['ay']),
                        float(payload['az']),
                        float(payload['gx']),
                        float(payload['gy']),
                        float(payload['gz'])
                    ]
                    data_chunk = [row] # Coloca dentro de uma lista para simular um "lote" de 1 item
                    print(f"Dado recebido: {row}")
                except KeyError as e:
                    print(f"Erro: Chave faltando no JSON: {e}")
                    return

            # CASO 2: Formato Lote (Original)
            elif 'data' in payload:
                data_chunk = payload['data']
                print(f"Dado recebido (Lote 'data'): {len(data_chunk)} linhas.")
            
            # CASO 3: Lista Direta
            elif isinstance(payload, list):
                data_chunk = payload
                print(f"Dado recebido (Lista direta): {len(data_chunk)} linhas.")
            
            else:
                print(f"Erro: JSON desconhecido. Chaves encontradas: {payload.keys()}")
                return 

            # Envia para o processamento
            self.ingest_data(data_chunk)

        except json.JSONDecodeError:
            print("Erro: Falha ao decodificar JSON.")
        except Exception as e:
            print(f"Erro genérico no processamento: {e}")

    def ingest_data(self, data_chunk):
        current_server_time = time.time()
        step = 1.0 / FREQUENCY
        
        # Validar dimensão dos dados (esperado: 6 colunas [ax,ay,az,gx,gy,gz])
        if len(data_chunk) > 0 and len(data_chunk[0]) != 6:
            print(f"Erro de Dimensão: Recebido {len(data_chunk[0])} colunas, esperado 6 (acc+gyro).")
            return

        for i, row in enumerate(data_chunk):
            estimated_ts = current_server_time - ((len(data_chunk) - 1 - i) * step)
            full_row = [estimated_ts] + row 
            self.buffer.append(full_row)
        
        # DEBUG 3: Status do Buffer
        # print(f"Buffer: {len(self.buffer)} / {WINDOW_SIZE} amostras necessárias para predição.")
        
        if len(self.buffer) >= WINDOW_SIZE:
            self._process_window()

    def _process_window(self):
        # DEBUG 4: Confirmar que a inferência está rodando
        # print("Iniciando inferência...") 
        
        window_list = list(self.buffer)[-WINDOW_SIZE:]
        window_array = np.array(window_list)

        timestamps = window_array[:, 0]
        sensor_data = window_array[:, 1:7].astype(float)

        try:
            feats = extract_features(sensor_data)
            df_feats = pd.DataFrame([feats])
            df_feats = df_feats.reindex(columns=self.features_esperadas, fill_value=0.0)
            
            X_scaled = self.scaler.transform(df_feats)
            
            # Tenta pegar score se possível para debug
            prediction = self.model.predict(X_scaled)[0]
            
            # Se o modelo tiver predict_proba ou decision_function, é útil printar
            # score = self.model.decision_function(X_scaled) if hasattr(self.model, "decision_function") else "N/A"

            if prediction == 1:
                print(f"[QUEDA] Detectada! (Features processadas OK)")
                now = time.time()
                if (now - self.last_fall_time) > COOLDOWN_SEC:
                    self.last_fall_time = now
                    self._publish_alert(timestamps[-1])
            else:
                # Opcional: printar que analisou e não é queda (pode poluir muito o log)
                #print("Normal (Não é queda).")
                pass

        except Exception as e:
            print(f"   ❌ Erro durante a inferência/extração de features: {e}")

    def _publish_alert(self, ts_evento):
        # Cria um JSON bonitinho para quem for ler (Dashboard/App)
        payload_dict = "FALL_DETECTED"
        payload_str = json.dumps(payload_dict)

        # --- CORREÇÃO DO TRAVAMENTO ---
        # Removido: infot.wait_for_publish()  <-- ISSO CAUSAVA O DEADLOCK
        
        # Apenas publica. A biblioteca Paho gerencia o envio em background.
        try:
            self.client.publish(TOPIC_OUTPUT, payload_str, qos=1)
            print(f"Alerta disparado para '{TOPIC_OUTPUT}' (Async)")
        except Exception as e:
            print(f"Falha ao tentar publicar: {e}")

if __name__ == '__main__':
    # Certifique-se que o .pkl está na mesma pasta!
    detector = MQTTFallDetector('detector_quedas_gmm_v3.pkl')
    detector.start()