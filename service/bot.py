import logging
import json
import asyncio
import ssl
from datetime import datetime

# Bibliotecas do Telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Biblioteca MQTT
import paho.mqtt.client as mqtt

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================
TELEGRAM_BOT_TOKEN = "8577563748:AAEEIAO66vR4Q-mc1KR7oB1UKepFT-ov0DY"

# Configurações do HiveMQ
MQTT_BROKER = "4f97c41ebc1e4119a1b005fe735cd462.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "escuta_mqtt"  # Usuário criado no HiveMQ
MQTT_PASSWORD = "trabalhoIOT1"
MQTT_TOPIC = "quedas"          # Tópico onde o script de detecção publica

# Configuração de Logs (Ajuda a ver erros)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
# Silencia logs excessivos de conexão HTTP
logging.getLogger("httpx").setLevel(logging.WARNING)

# ==============================================================================
# 2. CLASSE DO BOT TELEGRAM
# ==============================================================================
class FallDetectionBot:
    def __init__(self, token: str):
        self.token = token
        self.caregiver_chat_id = None
        self._event_loop = None  # Será preenchido quando o bot iniciar
        
        # O segredo está aqui: post_init
        self.application = (
            Application.builder()
            .token(token)
            .post_init(self.post_init) 
            .build()
        )
        
        # Comandos
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("set_cuidador", self.set_caregiver))
        self.application.add_handler(CommandHandler("status", self.status_command))

    async def post_init(self, application: Application):
        """Executado APÓS o bot criar o loop interno. Captura o loop correto."""
        self._event_loop = asyncio.get_running_loop()
        print("[BOT] Event Loop capturado com sucesso!")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        await update.message.reply_html(rf"Olá {user.mention_html()}!")
        await update.message.reply_text(
            "Detector de Quedas Ativado!\n\n"
            "IMPORTANTE: Digite /set_cuidador para receber os alertas aqui."
        )

    async def set_caregiver(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Define quem vai receber os alertas."""
        self.caregiver_chat_id = update.effective_chat.id
        await update.message.reply_text(
            f"Configurado! Você (ID: {self.caregiver_chat_id}) receberá os alertas de queda."
        )
        print(f"[BOT] Novo cuidador definido: {self.caregiver_chat_id}")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = "Bot Online\n"
        if self.caregiver_chat_id:
            msg += f"Cuidador Ativo: ID {self.caregiver_chat_id}"
        else:
            msg += "NENHUM Cuidador configurado! Use /set_cuidador"
        await update.message.reply_text(msg)

    async def send_fall_alert(self, message: str):
        """Função interna ASSÍNCRONA que envia a mensagem."""
        if not self.caregiver_chat_id:
            logging.warning("Queda detectada, mas nenhum cuidador cadastrado! Ignorando.")
            return

        try:
            timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            alert_text = (
                f"<b>QUEDA DETECTADA!</b>\n\n"
                f"ℹ{message}\n"
                f"{timestamp}\n\n"
                f"<i>Por favor, verifique o idoso imediatamente!</i>"
            )
            
            await self.application.bot.send_message(
                chat_id=self.caregiver_chat_id,
                text=alert_text,
                parse_mode="HTML"
            )
            logging.info(f"[BOT] Alerta enviado para {self.caregiver_chat_id}")
        except Exception as e:
            logging.error(f"[BOT] Erro ao enviar mensagem: {e}")

    def send_fall_alert_sync(self, message: str):
        """
        Ponte entre a Thread MQTT (Síncrona) e o Bot Telegram (Assíncrono).
        Agenda a tarefa no Event Loop principal.
        """
        if self._event_loop and self._event_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.send_fall_alert(message),
                self._event_loop
            )
        else:
            logging.error("[BOT] Erro Crítico: O Event Loop do Bot não está acessível.")

    def run(self):
        print("[BOT] Iniciando Polling do Telegram...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)


# ==============================================================================
# 3. CLASSE DO RECEPTOR MQTT
# ==============================================================================
class MQTTReceiver:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        
        # Configuração compatível com Paho v2
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        
        self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        
        # Configuração TLS para HiveMQ (Porta 8883)
        self.client.tls_set(
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLSv1_2
        )

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("[MQTT] Conectado ao HiveMQ!")
            client.subscribe(MQTT_TOPIC)
            print(f"[MQTT] Ouvindo tópico: '{MQTT_TOPIC}'")
        else:
            print(f"[MQTT] Falha na conexão. Código: {rc}")

    def on_message(self, client, userdata, msg):
        """Processa mensagens recebidas (Texto ou JSON)"""
        try:
            payload = msg.payload.decode()
            print(f"[MQTT] Recebido: {payload}")

            is_fall = False
            details = "Movimento brusco identificado."

            # 1. Tenta verificar se é JSON
            if "{" in payload and "}" in payload:
                try:
                    data = json.loads(payload)
                    # Verifica chaves típicas
                    if data.get("alerta") == "QUEDA_DETECTADA" or "FALL" in str(data):
                        is_fall = True
                        if "mensagem" in data:
                            details = data["mensagem"]
                except json.JSONDecodeError:
                    pass # Não era JSON válido, segue para texto
            
            # 2. Tenta verificar se é Texto Puro (Backward Compatibility)
            if not is_fall:
                if "FALL" in payload or "QUEDA" in payload:
                    is_fall = True

            # 3. Dispara o alerta se for queda
            if is_fall:
                print("[MQTT] Gatilho de queda acionado! Chamando Bot...")
                self.bot.send_fall_alert_sync(details)
            else:
                print("[MQTT] Mensagem ignorada (não parece alerta de queda).")

        except Exception as e:
            print(f"[MQTT] Erro ao processar mensagem: {e}")

    def start(self):
        print(f"[MQTT] Tentando conectar a {MQTT_BROKER}...")
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start() # Roda em background (Thread separada)
        except Exception as e:
            print(f"[MQTT] Erro fatal de conexão: {e}")


# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    # 1. Cria o Bot
    bot = FallDetectionBot(TELEGRAM_BOT_TOKEN)

    # 2. Cria o Receptor MQTT e conecta ao Bot
    mqtt_receiver = MQTTReceiver(bot)

    print("--- INICIANDO SISTEMA DE ALERTA ---")

    # 3. Inicia MQTT (Background)
    mqtt_receiver.start()

    # 4. Inicia Telegram (Bloqueia o terminal aqui)
    bot.run()

if __name__ == "__main__":
    main()