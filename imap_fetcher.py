# imap_fetcher.py
"""
IMAP fetcher for Smart Document Hub.

- Polls IMAP inbox (IMAP_HOST / IMAP_PORT / IMAP_USER / IMAP_PASS from .env)
- Looks for UNSEEN messages with attachments
- Saves attachments to UPLOAD_FOLDER
- Runs extraction, summary, classification, logging, forwarding, auto-reply
- Marks processed messages as SEEN
"""

import os
import time
import sqlite3
import imaplib
import email
from email.policy import default
from email.header import decode_header
from datetime import datetime
from dotenv import load_dotenv
import mimetypes
import smtplib
from email.message import EmailMessage

# Optional OCR / extraction libs — used if installed
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import docx
except Exception:
    docx = None

# Try to import OCR libs if available
try:
    from pdf2image import convert_from_path
    import pytesseract
except Exception:
    convert_from_path = None
    pytesseract = None

# load .env
load_dotenv()

# Config (expects keys in .env)
IMAP_HOST = os.getenv('IMAP_HOST')
IMAP_PORT = int(os.getenv('IMAP_PORT', '993'))
IMAP_USER = os.getenv('IMAP_USER')
IMAP_PASS = os.getenv('IMAP_PASS')

EMAIL_USER = os.getenv('EMAIL_USER')  # SMTP sender (bot)
EMAIL_PASS = os.getenv('EMAIL_PASS')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
DB_PATH = os.getenv('DB_PATH', 'history.db')

ROUTE_invoice = os.getenv('ROUTE_invoice')
ROUTE_payslip = os.getenv('ROUTE_payslip')
ROUTE_purchase_order = os.getenv('ROUTE_purchase_order')
ROUTE_minutes = os.getenv('ROUTE_minutes')

FROM_NAME = os.getenv('FROM_NAME', 'Smart Document Hub')
ADMIN_EMAIL = os.getenv('ADMIN_EMAIL')

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc'}

POLL_SECONDS = int(os.getenv('IMAP_POLL_SECONDS', '20'))  # how often to poll

# helpers (extraction, classify, log, send) - similar to app.py
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def simple_summarize(text, max_sentences=4):
    if not text:
        return ""
    # simple split fallback (avoid NLTK dependency here)
    parts = []
    for line in text.splitlines():
        for sent in line.split('.'):
            s = sent.strip()
            if s:
                parts.append(s + '.')
            if len(parts) >= max_sentences:
                break
        if len(parts) >= max_sentences:
            break
    return ' '.join(parts) if parts else (text[:300] + '...')

def extract_text(file_path, ocr_dpi=300, max_pages_for_ocr=30):
    if not file_path or not os.path.exists(file_path):
        return ""
    ext = file_path.rsplit('.', 1)[-1].lower()
    try:
        if ext == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        if ext == 'pdf':
            # try pdfplumber first if available
            if pdfplumber:
                try:
                    texts = []
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                texts.append(page_text)
                    combined = "\n".join(texts).strip()
                    if combined:
                        return combined
                except Exception as e:
                    print("pdfplumber error:", e)
            # fallback OCR if available
            if convert_from_path and pytesseract:
                try:
                    images = convert_from_path(file_path, dpi=ocr_dpi)
                except Exception as e:
                    print("pdf2image error:", e)
                    return ""
                ocr_texts = []
                for i, img in enumerate(images):
                    if i >= max_pages_for_ocr:
                        break
                    try:
                        txt = pytesseract.image_to_string(img)
                        if txt and txt.strip():
                            ocr_texts.append(txt)
                    except Exception as e:
                        print("pytesseract error on page", i, e)
                return "\n".join(ocr_texts).strip()
            # no extractor available
            return ""
        if ext in ('docx', 'doc'):
            if docx:
                try:
                    d = docx.Document(file_path)
                    paragraphs = [p.text for p in d.paragraphs if p.text and p.text.strip()]
                    return "\n".join(paragraphs).strip()
                except Exception as e:
                    print("docx extraction error:", e)
                    return ""
            else:
                return ""
    except Exception as e:
        print("extract_text error:", e)
        return ""
    return ""

def classify_document(text):
    t = (text or "").lower()
    invoice_keywords = ['invoice', 'amount due', 'invoice no', 'bill to', 'total', 'tax', 'gst']
    payslip_keywords = ['payslip', 'salary', 'net pay', 'pay period', 'gross pay']
    po_keywords = ['purchase order', 'po no']
    minutes_keywords = ['minutes of meeting', 'attendees', 'agenda', 'meeting']

    scores = {
        'invoice': sum(t.count(k) for k in invoice_keywords),
        'payslip': sum(t.count(k) for k in payslip_keywords),
        'purchase_order': sum(t.count(k) for k in po_keywords),
        'minutes': sum(t.count(k) for k in minutes_keywords)
    }

    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else 'other'

def log_upload(filename, saved_path, summary, category, uploader_email=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO uploads (filename, saved_path, summary, category, uploader_email, uploaded_at) VALUES (?, ?, ?, ?, ?, ?)',
              (filename, saved_path, summary, category, uploader_email, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def send_email_with_attachment(to_email, subject, body_text, attachment_path=None, attachment_name=None):
    if not EMAIL_USER or not EMAIL_PASS:
        print("SMTP not configured; cannot send email.")
        return False
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = f"{FROM_NAME} <{EMAIL_USER}>"
        msg['To'] = to_email
        msg.set_content(body_text)
        if attachment_path and os.path.exists(attachment_path):
            fname = attachment_name or os.path.basename(attachment_path)
            ctype, _ = mimetypes.guess_type(attachment_path)
            maintype, subtype = (ctype or 'application/octet-stream').split('/', 1)
            with open(attachment_path, 'rb') as f:
                msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=fname)
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print("SMTP send error:", e)
        return False

def route_for_category(category):
    mapping = {
        'invoice': ROUTE_invoice,
        'payslip': ROUTE_payslip,
        'purchase_order': ROUTE_purchase_order,
        'minutes': ROUTE_minutes
    }
    dest = mapping.get(category)
    return dest if dest and dest.strip() else None

# IMAP helper functions
def decode_mime_words(s):
    if not s:
        return ''
    decoded = []
    for part, enc in decode_header(s):
        if isinstance(part, bytes):
            try:
                decoded.append(part.decode(enc or 'utf-8', errors='ignore'))
            except Exception:
                decoded.append(part.decode('utf-8', errors='ignore'))
        else:
            decoded.append(part)
    return ''.join(decoded)

def save_attachment(part, dest_folder, filename_hint=None):
    filename = part.get_filename()
    if filename:
        filename = decode_mime_words(filename)
    else:
        filename = filename_hint or "attachment"
    # sanitize
    filename = filename.replace('/', '_').replace('\\', '_')
    # ensure unique
    ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    save_name = f"{ts}_{filename}"
    os.makedirs(dest_folder, exist_ok=True)
    full_path = os.path.join(dest_folder, save_name)
    with open(full_path, 'wb') as f:
        f.write(part.get_payload(decode=True))
    return full_path, save_name

import shutil   # add near other imports if not present

def process_saved_file(saved_path, orig_filename, sender_email=None):
    # run extraction -> summarize -> classify
    text = extract_text(saved_path)
    summary = simple_summarize(text) if text else "No text extracted."
    category = classify_document(text)

    # Move file into category folder
    cat_folder = (category if category else 'other')
    cat_folder = "".join(ch for ch in cat_folder if ch.isalnum() or ch in ('_', '-')).lower() or 'other'
    target_dir = os.path.join(UPLOAD_FOLDER, cat_folder)
    os.makedirs(target_dir, exist_ok=True)
    dest_path = os.path.join(target_dir, orig_filename)
    try:
        shutil.move(saved_path, dest_path)
        saved_path = dest_path
    except Exception as e:
        print("Warning: failed to move IMAP-saved file to category folder:", e)
        # keep saved_path as original if move fails

    # log into DB with updated saved_path
    log_upload(orig_filename, saved_path, summary, category, sender_email)

    # forwarding (same as before but using updated saved_path)
    forward_to = route_for_category(category)
    forwarded = False
    if forward_to:
        subj = f"[Auto-forward] {category.upper()} - {orig_filename}"
        body = f"File uploaded from mailbox and classified as '{category}'.\n\nSummary:\n{summary}"
        forwarded = send_email_with_attachment(forward_to, subj, body, saved_path, orig_filename)

    # auto-reply to sender (as before)
    auto_reply_sent = False
    if sender_email:
        reply_subject = f"Received: {orig_filename} (classified: {category})"
        reply_body = f"Hi,\n\nWe processed your attachment '{orig_filename}'.\nCategory: {category}\nSummary:\n{summary}\n\nThanks,\n{FROM_NAME}"
        auto_reply_sent = send_email_with_attachment(sender_email, reply_subject, reply_body, saved_path, orig_filename)

    print(f"Processed {orig_filename} -> category={category}, forwarded={forwarded}, auto-reply={auto_reply_sent}")


def process_message(msg, mail):
    """
    Process one email message: save attachments, process each file, then
    send a summary email to the original sender (and optionally ADMIN_EMAIL).
    """
    sender = msg.get('From')
    sender_email = None
    if sender:
        if '<' in sender and '>' in sender:
            start = sender.find('<') + 1
            end = sender.find('>')
            sender_email = sender[start:end].strip()
        else:
            sender_email = sender.strip()
    subject = decode_mime_words(msg.get('Subject'))
    print("Processing message:", subject, "from:", sender_email)

    # per-file result list for summary
    results = []  # list of dicts: {name, processed(bool), category, forwarded(bool), auto_reply(bool), note}

    total_attachments = 0
    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        if not filename:
            continue
        total_attachments += 1
        filename_decoded = decode_mime_words(filename)
        try:
            full_path, saved_name = save_attachment(part, UPLOAD_FOLDER, filename_hint=filename_decoded)
            if allowed_file(saved_name):
                # process and capture outcomes (we assume process_saved_file does printing; to capture booleans, replicate internals)
                text = extract_text(full_path)
                summary = simple_summarize(text) if text else "No text extracted."
                category = classify_document(text)
                # forward
                forward_to = route_for_category(category)
                forwarded = False
                if forward_to:
                    subj = f"[Auto-forward] {category.upper()} - {saved_name}"
                    body = f"File uploaded from mailbox and classified as '{category}'.\n\nSummary:\n{summary}"
                    forwarded = send_email_with_attachment(forward_to, subj, body, full_path, saved_name)
                # auto-reply to original sender for this file (existing behavior)
                auto_reply_sent = False
                if sender_email:
                    reply_subject = f"Received: {saved_name} (classified: {category})"
                    reply_body = f"Hi,\n\nWe processed your attachment '{saved_name}'.\nCategory: {category}\nSummary:\n{summary}\n\nThanks,\n{FROM_NAME}"
                    auto_reply_sent = send_email_with_attachment(sender_email, reply_subject, reply_body, None, None)
                # log
                log_upload(saved_name, full_path, summary, category, sender_email)
                # record result
                results.append({
                    'name': saved_name,
                    'processed': True,
                    'category': category,
                    'forwarded': bool(forwarded),
                    'auto_reply': bool(auto_reply_sent),
                    'note': ''
                })
                print(f"Processed {saved_name} -> category={category}, forwarded={forwarded}, auto-reply={auto_reply_sent}")
            else:
                note = "extension-not-allowed"
                results.append({
                    'name': saved_name,
                    'processed': False,
                    'category': None,
                    'forwarded': False,
                    'auto_reply': False,
                    'note': note
                })
                print("Attachment saved but extension not allowed:", saved_name)
        except Exception as e:
            results.append({
                'name': filename_decoded,
                'processed': False,
                'category': None,
                'forwarded': False,
                'auto_reply': False,
                'note': f"error: {e}"
            })
            print("Error handling attachment:", filename_decoded, e)

    # Prepare summary counts
    processed_success = sum(1 for r in results if r['processed'])
    processed_failed = sum(1 for r in results if not r['processed'])

    # Console summary
    if total_attachments > 0:
        print(f"📬 Summary: Message \"{subject}\" from {sender_email or 'unknown sender'}")
        print(f"   → {total_attachments} attachments total")
        print(f"   → {processed_success} processed successfully")
        print(f"   → {processed_failed} skipped/failed\n")
    else:
        print("No attachments found in this message.\n")

    # --- SEND summary email to sender (and admin if configured) ---
    # Build a readable summary body
    if total_attachments > 0:
        lines = []
        lines.append(f"Hello,")
        lines.append("")
        lines.append(f"We processed your email with subject: \"{subject}\"")
        lines.append(f"Received attachments: {total_attachments}")
        lines.append(f"Processed successfully: {processed_success}")
        lines.append(f"Skipped/failed: {processed_failed}")
        lines.append("")
        lines.append("Details per attachment:")
        for r in results:
            status = "OK" if r['processed'] else "SKIPPED/ERROR"
            cat = r['category'] or "-"
            fwd = "Yes" if r['forwarded'] else "No"
            ar = "Yes" if r['auto_reply'] else "No"
            note = f" ({r['note']})" if r.get('note') else ""
            lines.append(f" - {r['name']}: {status}; category={cat}; forwarded={fwd}; auto_reply={ar}{note}")
        lines.append("")
        lines.append("If you have questions, reply to this message.")
        body = "\n".join(lines)

        # send to sender
        if sender_email:
            subj = f"Smart Document Hub — Processing summary for your email: {subject}"
            sent = send_email_with_attachment(sender_email, subj, body, None, None)
            print(f"Summary email sent to sender {sender_email}: {sent}")

        # optionally send to admin as well
        if ADMIN_EMAIL and ADMIN_EMAIL.strip():
            admin_subj = f"[ADMIN] Processed email '{subject}' from {sender_email or 'unknown'}"
            admin_body = f"Admin summary:\n\n{body}"
            admin_sent = send_email_with_attachment(ADMIN_EMAIL, admin_subj, admin_body, None, None)
            print(f"Summary email sent to admin {ADMIN_EMAIL}: {admin_sent}")

    # return whether processed attachments were found
    return total_attachments > 0


    


def poll_imap_loop():
    if not IMAP_HOST or not IMAP_USER or not IMAP_PASS:
        print("IMAP credentials not configured. Set IMAP_HOST/IMAP_USER/IMAP_PASS in .env")
        return
    print("Starting IMAP poll loop. Poll interval:", POLL_SECONDS, "seconds")
    while True:
        try:
            mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
            mail.login(IMAP_USER, IMAP_PASS)
            mail.select('INBOX')
            # search unseen
            status, data = mail.search(None, '(UNSEEN)')
            if status != 'OK':
                print("IMAP search failed:", status, data)
                mail.logout()
                time.sleep(POLL_SECONDS)
                continue
            ids = data[0].split()
            if not ids:
                # nothing new
                mail.logout()
                time.sleep(POLL_SECONDS)
                continue
            print("Found", len(ids), "new messages")
            for num in ids:
                try:
                    status, msg_data = mail.fetch(num, '(RFC822)')
                    if status != 'OK':
                        print("Failed to fetch msg", num)
                        continue
                    raw = msg_data[0][1]
                    parsed = email.message_from_bytes(raw, policy=default)
                    processed = process_message(parsed, mail)
                    # mark as seen (even if no attachment) to avoid reprocessing
                    mail.store(num, '+FLAGS', '\\Seen')
                except Exception as e:
                    print("Error processing message", num, e)
            mail.logout()
        except Exception as e:
            print("IMAP connection error:", e)
        # wait for next poll
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    try:
        poll_imap_loop()
    except KeyboardInterrupt:
        print("Exiting IMAP fetcher.")
