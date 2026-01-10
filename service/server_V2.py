import joblib
import numpy as np
import pandas as pd
import json
import time
import ssl
import requests
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
TOPIC_INPUT = "data"
TOPIC_OUTPUT = "quedas"

# ==============================================================================
# CONFIGURAÇÃO API BACKEND
# ==============================================================================
API_URL = "http://localhost:8000/fall-registers/"

# ==============================================================================
# CONFIGURAÇÕES DE BUFFER
# ==============================================================================
FREQUENCY = 25.0
WINDOW_SECONDS = 3
WINDOW_SIZE = int(FREQUENCY * WINDOW_SECONDS)   # 75
BUFFER_MAX = WINDOW_SIZE * 2
COOLDOWN_SEC = 5

# ==============================================================================
# 2. CLASSE GMM (necessária para joblib)
# ==============================================================================
class GMMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components_0=1, n_components_1=1,
                 covariance_type_0='full', covariance_type_1='full',
                 threshold=0.0):
        self.n_components_0 = n_components_0
        self.n_components_1 = n_components_1
        self.covariance_type_0 = covariance_type_0
        self.covariance_type_1 = covariance_type_1
        self.threshold = threshold
        self.gmm_0 = None
        self.gmm_1 = None
        self.priors = None

    def fit(self, X, y):
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
# 3. EXTRAÇÃO DE FEATURES
# ==============================================================================
def extract_features(window):
    window = np.array(window)

    if window.size == 0 or window.ndim < 2 or window.shape[1] < 6:
        return None

    acc = window[:, :3]
    gyro = window[:, 3:]

    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    fs = FREQUENCY
    jerk = np.diff(acc_mag) * fs if len(acc_mag) > 1 else np.array([0.0])

    def safe_skew(x):
        try:
            v = skew(x)
            return 0.0 if np.isnan(v) else float(v)
        except:
            return 0.0

    def safe_kurtosis(x):
        try:
            v = kurtosis(x)
            return 0.0 if np.isnan(v) else float(v)
        except:
            return 0.0

    feats = {}

    acc_mean = np.mean(acc_mag)
    acc_max = np.max(acc_mag)
    acc_min = np.min(acc_mag)

    feats["acc_mag_mean"] = float(acc_mean)
    feats["acc_mag_std"] = float(np.std(acc_mag))
    feats["acc_mag_min"] = float(acc_min)
    feats["acc_mag_max"] = float(acc_max)
    feats["acc_mag_range"] = float(acc_max - acc_min)
    feats["acc_mag_p95"] = float(np.percentile(acc_mag, 95))
    feats["acc_mag_rms"] = float(np.sqrt(np.mean(acc_mag ** 2)))
    feats["acc_mag_skew"] = safe_skew(acc_mag)
    feats["acc_mag_kurtosis"] = safe_kurtosis(acc_mag)
    feats["acc_mag_sma"] = float(np.sum(np.abs(acc_mag - acc_mean)) / len(acc_mag))

    gyro_mean = np.mean(gyro_mag)
    gyro_max = np.max(gyro_mag)

    feats["gyro_mag_mean"] = float(gyro_mean)
    feats["gyro_mag_std"] = float(np.std(gyro_mag))
    feats["gyro_mag_max"] = float(gyro_max)
    feats["gyro_mag_range"] = float(gyro_max - np.min(gyro_mag))
    feats["gyro_mag_p95"] = float(np.percentile(gyro_mag, 95))
    feats["gyro_mag_rms"] = float(np.sqrt(np.mean(gyro_mag ** 2)))
    feats["gyro_mag_sma"] = float(np.sum(np.abs(gyro_mag - gyro_mean)) / len(gyro_mag))

    feats["jerk_mean"] = float(np.mean(np.abs(jerk)))
    feats["jerk_std"] = float(np.std(jerk))
    feats["jerk_max"] = float(np.max(np.abs(jerk)))

    n_post = int(len(acc_mag) * 0.3)
    post_mean = np.mean(acc_mag[-n_post:]) if n_post > 0 else 9.8
    feats["impact_ratio"] = float(acc_max / (post_mean + 1e-6))

    return feats

# ==============================================================================
# 4. SISTEMA MQTT WORKER
# ==============================================================================
class MQTTFallDetector:
    def __init__(self, model_file, scaler_file):
        print("[INIT] Carregando sistema...")

        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)

        self.features_esperadas = [
            "acc_mag_mean", "acc_mag_std", "acc_mag_min", "acc_mag_max",
            "acc_mag_range", "acc_mag_p95", "acc_mag_rms",
            "acc_mag_skew", "acc_mag_kurtosis", "acc_mag_sma",
            "gyro_mag_mean", "gyro_mag_std", "gyro_mag_max", "gyro_mag_range",
            "gyro_mag_p95", "gyro_mag_rms", "gyro_mag_sma",
            "jerk_mean", "jerk_std", "jerk_max", "impact_ratio"
        ]

        self.buffer = deque(maxlen=BUFFER_MAX)
        self.last_fall_time = 0
        self.msg_count = 0
        self.current_patient_id = None

        self.client = mqtt.Client()
        self.client.username_pw_set(MQTT_USER, MQTT_PASS)

        context = ssl.create_default_context()
        self.client.tls_set_context(context)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)

    def start(self):
        print("[LOOP] Detector rodando...")
        self.client.loop_forever()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(TOPIC_INPUT)
            print("[MQTT] Conectado e inscrito")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            self.msg_count += 1
            print(".", end="", flush=True)

            if isinstance(payload, dict) and "ax" in payload:
                self.current_patient_id = payload.get("patient_id")

                row = [
                    float(payload["ax"]),
                    float(payload["ay"]),
                    float(payload["az"]),
                    float(payload["gx"]),
                    float(payload["gy"]),
                    float(payload["gz"]),
                ]

                self.ingest_data([row])

        except Exception as e:
            print(f"\n[ERRO MQTT] {e}")

    def ingest_data(self, data_chunk):
        now = time.time()
        step = 1.0 / FREQUENCY

        for i, row in enumerate(data_chunk):
            ts = now - ((len(data_chunk) - 1 - i) * step)
            self.buffer.append([ts] + row)

        if len(self.buffer) >= WINDOW_SIZE:
            self._process_window()

    def _process_window(self):
        window = np.array(list(self.buffer)[-WINDOW_SIZE:])
        sensor_data = window[:, 1:7]

        feats = extract_features(sensor_data)
        if feats is None:
            return

        df = pd.DataFrame([feats])[self.features_esperadas]
        X = self.scaler.transform(df)

        score = self.model.predict_score(X)[0]
        is_fall = score > self.model.threshold

        if self.msg_count % 25 == 0:
            print(f"\n[ANALISE] Score={score:.3f} | {'QUEDA' if is_fall else 'Normal'}")

        if is_fall:
            now = time.time()
            if now - self.last_fall_time > COOLDOWN_SEC:
                self.last_fall_time = now
                print("\n🚨 QUEDA DETECTADA")
                self._publish_alert(window[-1][0])

    def _publish_alert(self, ts_evento):
        # MQTT (AGORA inclui patient_id)
        mqtt_payload = {
            "event": "FALL_DETECTED",
            "patient_id": self.current_patient_id,
            "timestamp": ts_evento
        }

        try:
            self.client.publish(
                TOPIC_OUTPUT,
                json.dumps(mqtt_payload),
                qos=1
            )
            print(f"\n[MQTT] Publicado em '{TOPIC_OUTPUT}': {mqtt_payload}")
        except Exception as e:
            print(f"\n[ERRO MQTT PUB] {e}")

        if not self.current_patient_id:
            print("Patient ID ausente")
            return

        try:
            data_points = []
            for row in list(self.buffer)[-WINDOW_SIZE:]:
                _, ax, ay, az, gx, gy, gz = row
                data_points.append({
                    "ax": ax, "ay": ay, "az": az,
                    "gx": gx, "gy": gy, "gz": gz
                })

            payload = {
                "patient_id": self.current_patient_id,
                "is_fall": True,
                "data_points": data_points
            }

            r = requests.post(API_URL, json=payload, timeout=5)
            print(f"API -> {r.status_code}")

        except Exception as e:
            print(f"[ERRO API] {e}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    MODEL_FILE = "gmm_classifier_V1.pkl"
    SCALER_FILE = "scaler_padrao.pkl"

    detector = MQTTFallDetector(MODEL_FILE, SCALER_FILE)
    detector.start()
