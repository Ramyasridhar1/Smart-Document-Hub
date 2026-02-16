# test_send.py — run with: python test_send.py
import os, smtplib
from email.message import EmailMessage

EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))

if not EMAIL_USER or not EMAIL_PASS:
    print("Missing EMAIL_USER or EMAIL_PASS in environment (.env must be loaded).")
    raise SystemExit(1)

msg = EmailMessage()
msg['Subject'] = "SDH test message"
msg['From'] = f"Smart Document Hub <{EMAIL_USER}>"
msg['To'] = EMAIL_USER
msg.set_content("This is a test message from Smart Document Hub (test_send.py)")

print("Attempting SMTP connect to", SMTP_SERVER, SMTP_PORT)
try:
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as smtp:
        smtp.set_debuglevel(1)
        smtp.starttls()
        smtp.login(EMAIL_USER, EMAIL_PASS)
        smtp.send_message(msg)
    print("Test email sent successfully to", EMAIL_USER)
except Exception as e:
    print("SMTP test failed:", repr(e))
