from fastapi import BackgroundTasks
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType

MAIL_CONF = ConnectionConfig(
    MAIL_USERNAME="safeiot2026@gmail.com",
    MAIL_PASSWORD="lqps uupd gusg eeqq",
    MAIL_FROM="safeiot2026@gmail.com",
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)

async def notify_observer_onboarding(observer: dict, background_tasks: BackgroundTasks):
    """
    Decide se e como notificar o observador.
    Hoje: e-mail com link do Telegram
    Amanhã: WhatsApp, push, etc.
    """
    if observer.get("telegram_chat_id"):
        return  # já onboardado

    await send_telegram_invite_email(observer, background_tasks)


async def send_telegram_invite_email(observer: dict, background_tasks: BackgroundTasks):
    observer_id = str(observer["_id"])
    email = observer["email"]
    name = observer.get("name", "Observador")

    telegram_link = f"https://t.me/safeiotbot?start=observer_{observer_id}"

    html = f"""
    <h2>Olá, {name}!</h2>
    <p>Para receber alertas de queda em tempo real:</p>
    <p><a href="{telegram_link}">👉 Ativar Telegram</a></p>
    """

    message = MessageSchema(
        subject="Ative alertas via Telegram – SafeIoT",
        recipients=[email],
        body=html,
        subtype=MessageType.html
    )

    fm = FastMail(MAIL_CONF)
    background_tasks.add_task(fm.send_message, message)
