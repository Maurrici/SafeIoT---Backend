import logging
import json
import asyncio
import ssl
import requests
from datetime import datetime

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters
)

import paho.mqtt.client as mqtt

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
TELEGRAM_BOT_TOKEN = "8577563748:AAEEIAO66vR4Q-mc1KR7oB1UKepFT-ov0DY"

MQTT_BROKER = "4f97c41ebc1e4119a1b005fe735cd462.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "escuta_mqtt"
MQTT_PASSWORD = "trabalhoIOT1"
MQTT_TOPIC = "quedas"

API_BASE = "http://localhost:8000"

logging.basicConfig(level=logging.INFO)

# ==============================================================================
# BOT TELEGRAM
# ==============================================================================
class FallDetectionBot:
    """
    Bot Telegram STATELESS para alertas de queda.

    - Alertas NÃO dependem de sessão
    - Login serve apenas para:
        • associar telegram_chat_id
        • permitir comandos informativos (/pacientes)
    """

    def __init__(self, token: str):
        self._event_loop = None

        # Sessão leve: chat_id -> observer_id
        self.sessions: dict[int, str] = {}

        self.application = (
            Application.builder()
            .token(token)
            .post_init(self.post_init)
            .build()
        )

        # Comandos
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("login", self.login))
        self.application.add_handler(CommandHandler("status", self.status))
        self.application.add_handler(CommandHandler("pacientes", self.list_patients))

        # Fallback
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.unknown_message)
        )

    async def post_init(self, app):
        self._event_loop = asyncio.get_running_loop()
        logging.info("[BOT] Event loop pronto")

    # --------------------------------------------------------------------------
    # COMANDOS
    # --------------------------------------------------------------------------
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 <b>Bot de Detecção de Quedas – SafeIoT</b>\n\n"
            "📌 Comandos disponíveis:\n"
            "/login email senha\n"
            "/status\n"
            "/pacientes\n\n"
            "ℹ️ O login é necessário apenas uma vez para ativar alertas.",
            parse_mode="HTML"
        )

    async def login(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if len(context.args) != 2:
            await update.message.reply_text("Uso correto:\n/login email senha")
            return

        email, password = context.args
        chat_id = update.effective_chat.id

        try:
            # --------------------------------------------------
            # LOGIN NO BACKEND
            # --------------------------------------------------
            r = requests.post(
                f"{API_BASE}/observers/login",
                json={"email": email, "password": password},
                timeout=5
            )

            if r.status_code != 200:
                await update.message.reply_text("❌ Email ou senha inválidos.")
                return

            observer = r.json()
            observer_id = observer["_id"]

            # --------------------------------------------------
            # DISTINGUE CADASTRO vs RE-LOGIN
            # --------------------------------------------------
            had_telegram_before = observer.get("telegram_chat_id") is not None

            # --------------------------------------------------
            # REGISTRA / ATUALIZA CHAT ID (idempotente)
            # --------------------------------------------------
            requests.post(
                f"{API_BASE}/observers/{observer_id}/telegram",
                json={"telegram_chat_id": chat_id},
                timeout=5
            )

            # Atualiza sessão local
            self.sessions[chat_id] = observer_id

            # --------------------------------------------------
            # MENSAGEM ADEQUADA
            # --------------------------------------------------
            if had_telegram_before:
                msg = (
                    "🔓 <b>Login realizado com sucesso!</b>\n\n"
                    "Agora você pode consultar seus pacientes usando:\n"
                    "/pacientes"
                )
            else:
                msg = (
                    "✅ <b>Cadastro realizado com sucesso!</b>\n\n"
                    f"👤 {observer['name']}\n"
                    f"📧 {observer['email']}\n\n"
                    "📡 Você receberá alertas automaticamente."
                )

            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            logging.error(e)
            await update.message.reply_text("❌ Erro ao conectar à API.")

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🟢 Bot operacional.\n"
            "ℹ️ Alertas funcionam mesmo após reinício."
        )

    async def list_patients(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id

        observer_id = self.sessions.get(chat_id)
        if not observer_id:
            await update.message.reply_text(
                "🔐 Para consultar pacientes, faça login:\n"
                "/login email senha"
            )
            return

        try:
            r = requests.get(
                f"{API_BASE}/patients",
                params={"observer_id": observer_id},
                timeout=5
            )

            if r.status_code != 200:
                await update.message.reply_text("❌ Erro ao buscar pacientes.")
                return

            patients = r.json()

            if not patients:
                await update.message.reply_text(
                    "ℹ️ Nenhum paciente associado a você."
                )
                return

            msg = "🧑‍⚕️ <b>Seus pacientes:</b>\n\n"
            for p in patients:
                msg += f"• {p['name']} (ID: {p['_id']})\n"

            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            logging.error(e)
            await update.message.reply_text("❌ Erro ao conectar à API.")

    async def unknown_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "❓ Comando não reconhecido.\n\n"
            "📌 Comandos disponíveis:\n"
            "/login email senha\n"
            "/status\n"
            "/pacientes"
        )

    # --------------------------------------------------------------------------
    # ALERTAS (STATELESS)
    # --------------------------------------------------------------------------
    async def send_alert(self, patient_id: str):
        try:
            # Busca dados do paciente
            patient_name = "Paciente desconhecido"
            r_patient = requests.get(
                f"{API_BASE}/patients/{patient_id}",
                timeout=5
            )

            if r_patient.status_code == 200:
                patient = r_patient.json()
                patient_name = patient.get("name", patient_name)

            # Busca observadores do paciente
            r = requests.get(
                f"{API_BASE}/observers",
                params={"patient_id": patient_id},
                timeout=5
            )

            if r.status_code != 200:
                return

            observers = r.json()
            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

            for obs in observers:
                chat_id = obs.get("telegram_chat_id")
                if not chat_id:
                    continue

                msg = (
                    "🚨 <b>QUEDA DETECTADA</b>\n\n"
                    f"🧑 <b>Paciente:</b> {patient_name}\n"
                    f"🆔 <b>ID:</b> {patient_id}\n"
                    f"🕒 {timestamp}\n\n"
                    "<i>Verifique imediatamente!</i>"
                )

                await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=msg,
                    parse_mode="HTML"
                )

                logging.info(
                    f"[BOT] Alerta enviado | paciente={patient_name} | chat_id={chat_id}"
                )

        except Exception as e:
            logging.error(e)

    def send_alert_sync(self, patient_id: str):
        if self._event_loop:
            asyncio.run_coroutine_threadsafe(
                self.send_alert(patient_id),
                self._event_loop
            )

    def run(self):
        self.application.run_polling()

# ==============================================================================
# MQTT RECEIVER
# ==============================================================================
class MQTTReceiver:
    def __init__(self, bot: FallDetectionBot):
        self.bot = bot
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

        self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.client.tls_set(
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLSv1_2
        )

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(MQTT_TOPIC, qos=1)
            logging.info(f"[MQTT] Inscrito em '{MQTT_TOPIC}'")

    def on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            if data.get("event") == "FALL_DETECTED":
                patient_id = data.get("patient_id")
                logging.info(f"[MQTT] FALL_DETECTED patient_id={patient_id}")
                self.bot.send_alert_sync(patient_id)
        except Exception as e:
            logging.error(e)

    def start(self):
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.client.loop_start()

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    bot = FallDetectionBot(TELEGRAM_BOT_TOKEN)
    mqtt = MQTTReceiver(bot)

    mqtt.start()
    bot.run()

if __name__ == "__main__":
    main()
