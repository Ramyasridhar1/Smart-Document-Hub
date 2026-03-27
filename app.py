# ---------------------- IMPORTS ----------------------
import os
import sqlite3
import time
import threading
import shutil
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, send_from_directory, abort
from werkzeug.utils import secure_filename, safe_join
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
import smtplib
from email.message import EmailMessage
import mimetypes
import pdfplumber
import docx
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import joblib
from math import ceil
import csv
from io import StringIO
from functools import wraps
from flask import session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import openai
import spacy
from settings_store import (
    DEFAULT_SETTING_KEYS,
    SENSITIVE_SETTING_KEYS,
    ensure_settings_schema,
    get_settings_bulk,
    get_settings_updated_at,
    set_setting,
)


# -----------------------------------------------------

# === NLTK local data bootstrap ===
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy NLP model loaded successfully")
except OSError:
    print("WARNING: spaCy model not found")
    nlp = None
# =====================================================

# ---------------- ML model loader -------------------
MODEL_PATH = os.path.join("model", "tfidf_logreg.joblib")
_ml_pipeline = None
try:
    if os.path.exists(MODEL_PATH):
        _ml_pipeline = joblib.load(MODEL_PATH)
        print(">>> ML classifier loaded from:", MODEL_PATH)
    else:
        print(">>> ML model not found at", MODEL_PATH)
except Exception as e:
    _ml_pipeline = None
    print(">>> Failed to load ML model:", e)
# ----------------------------------------------------

# ---------------------------------
# LOAD ENVIRONMENT AND CONFIG
# ---------------------------------
load_dotenv()


UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
DB_PATH = os.getenv('DB_PATH', 'history.db')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc'}

EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))

ROUTE_invoice = os.getenv('ROUTE_invoice')
ROUTE_payslip = os.getenv('ROUTE_payslip')
ROUTE_purchase_order = os.getenv('ROUTE_purchase_order')
ROUTE_minutes = os.getenv('ROUTE_minutes')

FROM_NAME = os.getenv('FROM_NAME', 'Smart Document Hub')
ADMIN_EMAIL = os.getenv('ADMIN_EMAIL')

# ----------------- AUTH CONFIG -----------------
ADMIN_USER = os.getenv('ADMIN_USER', 'admin')
ADMIN_PASS = os.getenv('ADMIN_PASS', None)
# -----------------------------------------------

# ---------------------------------
# FLASK APP INIT
# ---------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB

SECRET_KEY = os.getenv('FLASK_SECRET_KEY', None) or os.getenv('SECRET_KEY', None) or os.urandom(24).hex()
app.secret_key = SECRET_KEY
# expose a few helpful vars to all templates so templates can use them directly
@app.context_processor
def inject_template_globals():
    runtime = get_runtime_settings()
    try:
        cfg = app.config
    except Exception:
        cfg = {}
    cfg['EMAIL_USER'] = runtime.get('email_user')
    cfg['ROUTE_invoice'] = runtime.get('route_invoice')
    cfg['ROUTE_payslip'] = runtime.get('route_payslip')
    cfg['ROUTE_purchase_order'] = runtime.get('route_purchase_order')
    cfg['ROUTE_minutes'] = runtime.get('route_minutes')
    cfg['ADMIN_USER'] = ADMIN_USER
    return {
        'app': app,
        'config': cfg,
        'EMAIL_USER': runtime.get('email_user'),
        'ROUTE_invoice': runtime.get('route_invoice'),
        'ROUTE_payslip': runtime.get('route_payslip'),
        'ROUTE_purchase_order': runtime.get('route_purchase_order'),
        'ROUTE_minutes': runtime.get('route_minutes'),
        'FROM_NAME': runtime.get('from_name'),
    }

if ADMIN_PASS:
    _ADMIN_PASS_HASH = generate_password_hash(ADMIN_PASS)
else:
    _ADMIN_PASS_HASH = generate_password_hash("changeme")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login', next=request.path))
        if session.get('username') != ADMIN_USER:
            return abort(403)
        return f(*args, **kwargs)
    return decorated_function


def _is_valid_email(value):
    if not value:
        return False
    if ' ' in value:
        return False
    return '@' in value and '.' in value.split('@')[-1]


def _safe_int(value, default):
    try:
        return int(str(value).strip())
    except Exception:
        return default


def get_runtime_settings():
    settings = {
        'email_user': EMAIL_USER,
        'email_pass': EMAIL_PASS,
        'smtp_server': SMTP_SERVER,
        'smtp_port': str(SMTP_PORT),
        'imap_host': os.getenv('IMAP_HOST'),
        'imap_port': os.getenv('IMAP_PORT', '993'),
        'imap_user': os.getenv('IMAP_USER'),
        'imap_pass': os.getenv('IMAP_PASS'),
        'route_invoice': ROUTE_invoice,
        'route_payslip': ROUTE_payslip,
        'route_purchase_order': ROUTE_purchase_order,
        'route_minutes': ROUTE_minutes,
        'from_name': FROM_NAME,
        'admin_email': ADMIN_EMAIL,
        'imap_poll_seconds': os.getenv('IMAP_POLL_SECONDS', '20'),
    }

    try:
        db_values = get_settings_bulk(DB_PATH)
        for key in DEFAULT_SETTING_KEYS:
            if key in db_values and db_values.get(key) is not None:
                settings[key] = db_values.get(key)
    except Exception:
        pass

    settings['smtp_port'] = _safe_int(settings.get('smtp_port'), 587)
    settings['imap_port'] = _safe_int(settings.get('imap_port'), 993)
    settings['imap_poll_seconds'] = _safe_int(settings.get('imap_poll_seconds'), 20)
    return settings


def audit_log(action, details=''):
    actor = session.get('username') if session else None
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute(
            'INSERT INTO audit_log (actor, action, details, created_at) VALUES (?, ?, ?, ?)',
            (actor or 'system', action, details, datetime.utcnow().isoformat()),
        )
        conn.commit()
    except Exception as e:
        print('Audit log failed:', e)
    finally:
        conn.close()

# ---------------------------------
# DATABASE INIT
# ---------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS uploads (
                    id INTEGER PRIMARY KEY,
                    filename TEXT,
                    summary TEXT,
                    category TEXT,
                    uploader_email TEXT,
                    uploaded_at TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT,
                    role TEXT,
                    message TEXT,
                    created_at TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    is_encrypted INTEGER DEFAULT 0,
                    updated_at TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY,
                    actor TEXT,
                    action TEXT,
                    details TEXT,
                    created_at TEXT
                )''')
    conn.commit()

    cols = [r[1] for r in c.execute("PRAGMA table_info(uploads)").fetchall()]
    if 'saved_path' not in cols:
        try:
            c.execute("ALTER TABLE uploads ADD COLUMN saved_path TEXT")
            print("Added 'saved_path' column to uploads table.")
        except Exception as e:
            print("DB migration error:", e)
    conn.commit()
    conn.close()
    ensure_settings_schema(DB_PATH)

# ---------------------------------
# HELPERS
# ---------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def simple_summarize(text, max_sentences=4):
    if not text or not nlp:
        return ""
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return ' '.join(sentences[:max_sentences])


def extract_text(file_path, ocr_dpi=300, max_pages_for_ocr=50):
    if not file_path or not os.path.exists(file_path):
        return ""

    ext = file_path.rsplit('.', 1)[-1].lower()

    try:
        if ext == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        if ext == 'pdf':
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

        if ext in ('docx', 'doc'):
            try:
                document = docx.Document(file_path)
                paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
                return "\n".join(paragraphs).strip()
            except Exception as e:
                print("docx error:", e)
                return ""

    except Exception as e:
        print("extract_text general error:", e)
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
    c.execute(
        'INSERT INTO uploads (filename, saved_path, summary, category, uploader_email, uploaded_at) VALUES (?, ?, ?, ?, ?, ?)',
        (filename, saved_path, summary, category, uploader_email, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def send_email_with_attachment(to_email, subject, body_text, attachment_path=None, attachment_name=None):
    """
    Send one email with an optional single attachment.
    Returns True on success, False on failure.
    """
    runtime = get_runtime_settings()
    smtp_user = runtime.get('email_user')
    smtp_pass = runtime.get('email_pass')
    from_name = runtime.get('from_name') or FROM_NAME
    smtp_server = runtime.get('smtp_server') or SMTP_SERVER
    smtp_port = _safe_int(runtime.get('smtp_port'), 587)

    if not smtp_user or not smtp_pass:
        print("SMTP credentials missing.")
        return False
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = f"{from_name} <{smtp_user}>"
        msg['To'] = to_email
        msg.set_content(body_text)

        if attachment_path and os.path.exists(attachment_path):
            fname = attachment_name or os.path.basename(attachment_path)
            ctype, _ = mimetypes.guess_type(attachment_path)
            maintype, subtype = (ctype or 'application/octet-stream').split('/', 1)
            with open(attachment_path, 'rb') as f:
                msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=fname)

        with smtplib.SMTP(smtp_server, smtp_port) as smtp:
            smtp.starttls()
            smtp.login(smtp_user, smtp_pass)
            smtp.send_message(msg)
        print(f"📨 Sent single email with subject: {subject} to {to_email}")
        return True
    except Exception as e:
        print("Email send error:", e)
        return False


def send_email_with_attachments(to_email, subject, body_text, attachment_paths=None, attachment_names=None):
    """
    Send an email (SMTP) with multiple attachments.
    - attachment_paths: list of file paths to attach (can be empty or None)
    - attachment_names: optional list of filenames to use for attachments (same length as attachment_paths)
    Returns True on success, False on failure.
    """
    runtime = get_runtime_settings()
    smtp_user = runtime.get('email_user')
    smtp_pass = runtime.get('email_pass')
    from_name = runtime.get('from_name') or FROM_NAME
    smtp_server = runtime.get('smtp_server') or SMTP_SERVER
    smtp_port = _safe_int(runtime.get('smtp_port'), 587)

    if not smtp_user or not smtp_pass:
        print("SMTP credentials missing.")
        return False
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = f"{from_name} <{smtp_user}>"
        msg['To'] = to_email
        msg.set_content(body_text)

        if attachment_paths:
            for idx, ap in enumerate(attachment_paths):
                try:
                    if not ap or not os.path.exists(ap):
                        print(f"⚠️ Attachment missing, skipping: {ap}")
                        continue
                    fname = None
                    if attachment_names and idx < len(attachment_names):
                        fname = attachment_names[idx]
                    fname = fname or os.path.basename(ap)
                    ctype, _ = mimetypes.guess_type(ap)
                    maintype, subtype = (ctype or 'application/octet-stream').split('/', 1)
                    with open(ap, 'rb') as f:
                        msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=fname)
                    print(f"📎 Attached: {fname}")
                except Exception as e:
                    print(f"Failed to attach {ap}: {e}")

        with smtplib.SMTP(smtp_server, smtp_port) as smtp:
            smtp.starttls()
            smtp.login(smtp_user, smtp_pass)
            smtp.send_message(msg)
        print(f"✅ Email with {len(attachment_paths or [])} attachments sent to {to_email}")
        return True
    except Exception as e:
        print("Email send error (multiple attachments):", e)
        return False

def route_for_category(category):
    runtime = get_runtime_settings()
    mapping = {
        'invoice': runtime.get('route_invoice'),
        'payslip': runtime.get('route_payslip'),
        'purchase_order': runtime.get('route_purchase_order'),
        'minutes': runtime.get('route_minutes')
    }
    dest = mapping.get(category)
    return dest if dest and dest.strip() else None

# ---------------------------------
# ROUTES
# ---------------------------------
@app.route('/')
@login_required
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('file')
    uploader_email = request.form.get('email', None)

    if not files or all(f.filename == '' for f in files):
        return redirect(request.url)

    results = []  # collect all results for the batch

    for file in files:
        if not (file and allowed_file(file.filename)):
            continue

        filename = secure_filename(file.filename)
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(saved_path)

        # Extract, summarize, classify
        text = extract_text(saved_path)
        summary = simple_summarize(text) if text else "No text extracted."
        category = classify_document(text)

        # Create category folder and move file there
        cat_folder = (category if category else 'other')
        cat_folder = "".join(ch for ch in cat_folder if ch.isalnum() or ch in ('_', '-')).lower() or 'other'
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], cat_folder)
        os.makedirs(target_dir, exist_ok=True)
        dest_path = os.path.join(target_dir, saved_filename)
        try:
            shutil.move(saved_path, dest_path)
            saved_path = dest_path
        except Exception as e:
            print("⚠️ Warning: failed to move file to category folder:", e)
            # saved_path remains original if move fails

        # Log upload
        log_upload(saved_filename, saved_path, summary, category, uploader_email)

        # NOTE: per-file forwarding removed here (we do batch forwarding after the loop)

        # record result details for later (including route target if configured)
        forward_to = route_for_category(category)
        results.append({
            'filename': saved_filename,
            'category': category,
            'summary': summary,
            'route': forward_to,
            'saved_path': saved_path,
            'uploader_email': uploader_email
        })

    if not results:
        return "No valid files uploaded.", 400

    # ------------------ BATCH FORWARDING (group by recipient) ------------------
    from collections import defaultdict
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    batch_map = defaultdict(list)

    # Group items by recipient (route field)
    for r in results:
        target = r.get('route')
        if target:
            batch_map[target].append(r)

    # For each recipient, create an in-memory CSV, save temporarily, send once, then delete
    batch_sent_info = []
    for recipient, items in batch_map.items():
        # build CSV summary
        si = StringIO()
        writer = csv.writer(si)
        writer.writerow(['filename', 'category', 'summary', 'saved_path'])
        for it in items:
            writer.writerow([
                it.get('filename'),
                it.get('category') or '',
                (it.get('summary') or '').replace('\n', ' ')[:2000],
                it.get('saved_path') or ''
            ])
        csv_text = si.getvalue()
        si.close()

        csv_name = f"batch_summary_{timestamp}_{recipient.replace('@','_at_').replace('.','_')}.csv"
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_name)
        try:
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(csv_text)
        except Exception as e:
            print("Failed to write batch CSV:", e)
            continue

        # prepare attachments: actual files + the CSV summary
        attachment_paths = []
        attachment_names = []

        # Attach actual files (only those that exist)
        for it in items:
            p = it.get('saved_path')
            if p and os.path.exists(p) and os.path.isfile(p):
                attachment_paths.append(p)
                attachment_names.append(os.path.basename(p))
            else:
                print(f"⚠️ Skipping missing file for attachment: {p}")

        # Attach the CSV summary as last attachment
        attachment_paths.append(csv_path)
        attachment_names.append(os.path.basename(csv_path))

        subj = f"[Batch Forwarded Files] {len(items)} files - {timestamp}"
        body_lines = [
            "Hello,",
            "",
            f"This is an automated batch of files classified and routed to you.",
            "",
            "Files included:"
        ]
        for it in items:
            body_lines.append(f"- {it.get('filename')}  (category: {it.get('category')})")
            first_line = (it.get('summary') or "").splitlines()[0] if it.get('summary') else ''
            if first_line:
                body_lines.append(f"    summary: {first_line[:200]}")
        body_lines.extend(["", "Both the original files and a CSV summary are attached."])
        body = "\n".join(body_lines)

        # send actual files + csv in one email
        print(f"📤 Sending {len(attachment_paths)} attachments to {recipient} ...")
        sent = send_email_with_attachments(recipient, subj, body, attachment_paths=attachment_paths, attachment_names=attachment_names)
        batch_sent_info.append((recipient, sent))
        print(f"📨 send_email_with_attachments returned {sent} for {recipient}")

        # cleanup temp csv (we do not remove original uploaded files)
        try:
            if os.path.exists(csv_path):
                os.remove(csv_path)
                print(f"🧹 Deleted temporary CSV: {csv_path}")
        except Exception as e:
            print("⚠️ Failed to delete temp CSV:", e)
    # -------------------------------------------------------------------------

    # Optional: send auto-replies to uploaders (one per uploader)
    # This preserves the auto-reply behavior but sends only one reply per uploader/file
    for r in results:
        uploader = r.get('uploader_email')
        if uploader:
            runtime = get_runtime_settings()
            from_name = runtime.get('from_name') or FROM_NAME
            reply_subject = f"Receipt: {r.get('filename')} (classified: {r.get('category')})"
            reply_body = f"Hi,\n\nWe processed your file '{r.get('filename')}'.\nCategory: {r.get('category')}\nSummary:\n{r.get('summary')}\n\nThanks,\n{from_name}"
            try:
                send_email_with_attachment(uploader, reply_subject, reply_body, None, None)
            except Exception as e:
                print("Auto-reply failed for", uploader, e)

    return render_template('result_batch.html', results=results)

@app.route('/chat')
@login_required
def chat_page():
    return render_template('chat.html')


import requests
import json

import requests
import os
from datetime import datetime
from nltk.tokenize import sent_tokenize

import csv
from io import StringIO

@app.route('/chat/send', methods=['POST'])
@login_required
def chat_send():
    """
    Enhanced rule-based chat endpoint with:
      - FAQ + pattern matching
      - DB-connected answers (counts, last uploads)
      - Action triggers: forward by category, export CSV
      - Multi-step dialogs stored in session['pending_action']
      - Memory for user name in session['user_name']
      - Logs both user and assistant messages to chats table
    """
    data = request.get_json() or {}
    session_id = data.get('session_id', 'default')
    raw_prompt = (data.get('prompt') or '').strip()

    if not raw_prompt:
        return jsonify({'reply': 'Please send a non-empty prompt.'}), 400

    prompt = raw_prompt.strip()
    lower = prompt.lower()

    # Persist user message to chats table
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO chats (session_id, role, message, created_at) VALUES (?, ?, ?, ?)',
                  (session_id, 'user', prompt, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        print("DB write (user) failed:", e)

    # Initialize session state containers
    if 'pending_action' not in session:
        session['pending_action'] = None
    if 'user_name' not in session:
        session['user_name'] = None

    # Helper: execute a simple DB query and return rows
    def db_query(sql, params=()):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(sql, params)
            rows = c.fetchall()
            conn.close()
            return rows
        except Exception as e:
            print("DB query error:", e)
            return []

    # Helper: count files by category or overall
    def count_category(cat=None):
        if cat:
            r = db_query("SELECT COUNT(*) FROM uploads WHERE category = ?", (cat,))
        else:
            r = db_query("SELECT COUNT(*) FROM uploads")
        return r[0][0] if r else 0

    # Helper: last N uploads (id, filename, category, uploaded_at)
    def last_uploads(n=5, category=None):
        if category:
            rows = db_query("SELECT id, filename, category, uploaded_at FROM uploads WHERE category = ? ORDER BY id DESC LIMIT ?", (category, n))
        else:
            rows = db_query("SELECT id, filename, category, uploaded_at FROM uploads ORDER BY id DESC LIMIT ?", (n,))
        return rows

    # Helper: forward files for a given category to configured route (reuses send_email_with_attachment)
    def forward_files_to_route(category, recipient):
        # fetch saved_paths for last X matching files (we forward up to 10 newest to keep payload sane)
        files = db_query("SELECT filename, saved_path FROM uploads WHERE category = ? ORDER BY id DESC LIMIT 10", (category,))
        if not files:
            return False, "No recently uploaded files found for that category."
        sent_any = False
        failures = []
        for fname, path in files:
            if not path or not os.path.exists(path):
                failures.append(fname + " (missing)")
                continue
            subj = f"[Forwarded by SDH] {category.upper()} - {fname}"
            body = f"Forwarding file {fname} classified as {category}."
            ok = send_email_with_attachment(recipient, subj, body, path, fname)
            if ok:
                sent_any = True
            else:
                failures.append(fname)
        msg = "Forwarded files." if sent_any else "No files forwarded."
        if failures:
            msg += " Failures: " + ", ".join(failures)
        return sent_any, msg

    # Helper: export a CSV summary of last N uploads for a category OR all
    def export_summary_csv(category=None, n=100):
        if category:
            rows = db_query("SELECT filename, category, uploader_email, uploaded_at, saved_path, summary FROM uploads WHERE category = ? ORDER BY id DESC LIMIT ?", (category, n))
        else:
            rows = db_query("SELECT filename, category, uploader_email, uploaded_at, saved_path, summary FROM uploads ORDER BY id DESC LIMIT ?", (n,))
        if not rows:
            return None, "No rows to export."
        si = StringIO()
        wr = csv.writer(si)
        wr.writerow(['filename','category','uploader_email','uploaded_at','saved_path','summary'])
        for r in rows:
            wr.writerow([r[0], r[1] or '', r[2] or '', r[3] or '', r[4] or '', (r[5] or '').replace('\n',' ')[:3000]])
        csv_text = si.getvalue()
        si.close()
        fname = f"sdh_export_{(category or 'all')}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        try:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write(csv_text)
            return path, None
        except Exception as e:
            print("CSV write failed:", e)
            return None, "Failed to write CSV."

    # If there's a pending multi-step action, handle confirmations/next step
    pending = session.get('pending_action')
    if pending:
        # Example pending shapes:
        # {'type':'forward_confirm','category':'invoice','recipient':'x@y.com'}
        pt = pending.get('type')
        if pt == 'forward_confirm':
            # expecting yes/no
            if lower in ('yes','y','sure','ok','please do','do it'):
                category = pending.get('category')
                recipient = pending.get('recipient')
                ok, msg = forward_files_to_route(category, recipient)
                session['pending_action'] = None
                reply_text = f"{msg}"
            else:
                session['pending_action'] = None
                reply_text = "Okay — cancelled forwarding."
            # persist assistant reply
            try:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute('INSERT INTO chats (session_id, role, message, created_at) VALUES (?, ?, ?, ?)',
                          (session_id, 'assistant', reply_text, datetime.utcnow().isoformat()))
                conn.commit()
                conn.close()
            except Exception as e:
                print("DB write (assistant) failed:", e)
            return jsonify({'reply': reply_text})

        if pt == 'export_confirm':
            if lower in ('yes','y','ok','export','please'):
                category = pending.get('category')  # may be None for all
                path, err = export_summary_csv(category=category, n=500)
                session['pending_action'] = None
                if path:
                    # if admin email configured, attach and send
                    admin = ADMIN_EMAIL or os.getenv('ADMIN_EMAIL')
                    runtime = get_runtime_settings()
                    admin = runtime.get('admin_email') or admin
                    if admin:
                        send_email_with_attachment(admin, f"Export: {category or 'all'}", "Attached CSV export.", path, os.path.basename(path))
                    reply_text = f"CSV export created and saved to {path}."
                else:
                    reply_text = f"Export failed: {err}"
            else:
                session['pending_action'] = None
                reply_text = "Export cancelled."
            # persist assistant reply
            try:
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute('INSERT INTO chats (session_id, role, message, created_at) VALUES (?, ?, ?, ?)',
                          (session_id, 'assistant', reply_text, datetime.utcnow().isoformat()))
                conn.commit()
                conn.close()
            except Exception as e:
                print("DB write (assistant) failed:", e)
            return jsonify({'reply': reply_text})

    # No pending action — normal processing
    reply_text = None

    # 1) Name capture: "my name is X" or "call me X"
    if any(phrase in lower for phrase in ("my name is ", "call me ")):
        import re
        m = re.search(r"(?:my name is|call me)\s+([A-Za-z0-9 _-]{1,40})", prompt, re.I)
        if m:
            name = m.group(1).strip()
            session['user_name'] = name
            reply_text = f"Nice to meet you, {name}! I will remember that during this session."
        else:
            reply_text = "I didn't catch the name — please say, for example, 'Call me Ramya'."

    # 2) FAQ / simple intents
    faq_map = {
        # 👋 Greetings & Basic Help
        'hello': "Hey there 👋! I’m your Smart Document Hub Assistant. I can help you manage and classify your files automatically.",
        'hi': "Hi! How’s it going? You can ask me to show your uploads, count invoices, or even export a CSV summary.",
        'hey': "Hey! 😊 Ready to organize your documents today?",
        'help': (
            "Here’s what I can do for you:\n"
            "• Upload & auto-classify files (Invoice, Payslip, etc.)\n"
            "• Show your recent uploads\n"
            "• Forward files by category\n"
            "• Export CSV reports\n"
            "• Tell you stats like 'How many invoices?'\n"
            "Try typing: 'Show last 5 uploads' or 'Forward payslips'."
        ),
        'what can you do': "I help you organize, summarize, and forward your business documents automatically.",

        # 📂 Uploads & Processing
        'upload': (
            "To upload a file, go to the Home page and click 'Upload'. "
            "I'll extract text, summarize it, classify it (Invoice, Payslip, etc.), and store it neatly."
        ),
        'how to upload': "Click the 'Upload' button on the dashboard and select one or more files.",
        'multiple files': "Yes, you can upload multiple files at once! I’ll classify each automatically.",
        'formats': "I currently support PDF, DOCX, TXT, and PPTX files. You can also upload scanned PDFs — I’ll read them using OCR.",

        # 🧠 Classification
        'classify': "I automatically classify files into Invoice, Payslip, Purchase Order, or Minutes of Meeting based on their content.",
        'categories': "I currently recognize 4 categories: invoices, payslips, purchase orders, and meeting minutes.",
        'add category': "For now, categories are fixed, but we can train a model or add new rules to expand classification.",
        'ppt': "Yes, I can now read and classify PPT and PPTX files too!",

        # 🔎 History & Reports
        'history': (
            "Open the History page to view all uploads with filters for category, date, and keywords. "
            "You can also export them as CSV."
        ),
        'export': (
            "You can type 'Export invoices' or 'Export all' to get a CSV file of recent uploads. "
            "I’ll even email it to the admin if configured."
        ),
        'report': "I can export a detailed CSV report of your classified documents. Try saying: 'Export payslips'.",

        # ✉️ Email & Forwarding
        'forward': (
            "You can ask me to forward documents automatically! For example, type 'Forward invoices'. "
            "I'll send the latest ones to the configured email."
        ),
        'auto reply': "Yes, every uploader receives a confirmation email once their document is processed successfully.",
        'email setup': (
            "All outgoing and incoming emails are managed using Gmail’s SMTP and IMAP via your environment configuration."
        ),

        # 🧾 Stats
        'how many': None,  # handled dynamically
        'count': None,
        'total': None,
        'summary': "I summarize each file briefly when you upload it — it helps identify the key content quickly.",

        # ⚙️ Settings
        'routes': (
            f"Here are the configured routing emails:\n"
            f"• Invoices → {route_for_category('invoice') or 'not set'}\n"
            f"• Payslips → {route_for_category('payslip') or 'not set'}\n"
            f"• Purchase Orders → {route_for_category('purchase_order') or 'not set'}\n"
            f"• Minutes → {route_for_category('minutes') or 'not set'}"
        ),
        'change route': (
            "To change routes and mailbox credentials, open Admin Settings from the dashboard."
        ),

        # 🧑 Personal
        'who made you': "I was created by Ramya S, as part of the Smart Document Hub project 💻.",
        'your name': "You can call me DocuBot 🤖 — your document assistant.",
        'bye': "Goodbye! 👋 Have a productive day ahead!",
        'thanks': "You're very welcome, Ramya! 😊",
        'thank you': "Anytime! Always happy to help!",
    }

    # direct exact keyword check (small)
    for k, v in faq_map.items():
        if k in lower and v:
            reply_text = v
            break

    # 3) Count queries (e.g., "how many invoices", "count invoices")
    if not reply_text and any(w in lower for w in ('how many', 'count', 'number of')):
        for cat_keyword, cat_name in [('invoice','invoice'), ('payslip','payslip'), ('purchase','purchase_order'), ('minutes','minutes')]:
            if cat_keyword in lower:
                cnt = count_category(cat_name)
                reply_text = f"There are {cnt} files classified as '{cat_name}'."
                break
        if not reply_text and 'files' in lower:
            total = count_category(None)
            reply_text = f"Total uploaded files: {total}."

    # 4) Last uploads (e.g., "show last 5 uploads", "recent invoices")
    if not reply_text and any(kw in lower for kw in ('last', 'recent', 'show')) and ('upload' in lower or 'uploads' in lower or 'recent' in lower):
        import re
        m = re.search(r'last\s+(\d{1,2})', lower)
        n = int(m.group(1)) if m else 5
        # optional category detection
        category = None
        for cat_keyword, cat_name in [('invoice','invoice'), ('payslip','payslip'), ('purchase','purchase_order'), ('minutes','minutes')]:
            if cat_keyword in lower:
                category = cat_name
                break
        rows = last_uploads(n, category)
        if not rows:
            reply_text = "No uploads found matching that query."
        else:
            out_lines = []
            for rid, fname, cat, uploaded_at in rows:
                out_lines.append(f"{fname} ({cat}) — {uploaded_at or 'time unknown'}")
            reply_text = "Recent uploads:\n" + "\n".join(out_lines[:n])

    # 5) Forward command (e.g., "forward invoices", "please forward invoices")
    if not reply_text and 'forward' in lower:
        # find category
        category = None
        for cat_keyword, cat_name in [('invoice','invoice'), ('payslip','payslip'), ('purchase','purchase_order'), ('minutes','minutes')]:
            if cat_keyword in lower:
                category = cat_name
                break
        if not category:
            reply_text = "Which category do you want to forward? (e.g., invoices, payslips, purchase orders, minutes)"
        else:
            recipient = route_for_category(category)
            if not recipient:
                reply_text = f"No route configured for {category}. Update it in Admin Settings."
            else:
                # ask for confirmation as a multi-step action
                session['pending_action'] = {'type':'forward_confirm','category':category,'recipient':recipient}
                reply_text = f"Do you want me to forward the recent files in category '{category}' to {recipient}? Reply 'yes' to confirm."

    # 6) Export command (e.g., "export invoices", "export csv")
    if not reply_text and ('export' in lower or 'download csv' in lower or 'export csv' in lower):
        # optional category
        category = None
        for cat_keyword, cat_name in [('invoice','invoice'), ('payslip','payslip'), ('purchase','purchase_order'), ('minutes','minutes')]:
            if cat_keyword in lower:
                category = cat_name
                break
        session['pending_action'] = {'type':'export_confirm','category':category}
        reply_text = f"Okay — I'll create a CSV export for {category or 'all categories'}. Reply 'yes' to proceed."

    # 7) Misc: name greeting, or default local reply
    if not reply_text:
        if session.get('user_name'):
            reply_text = f"Yes {session.get('user_name')} — I heard: \"{prompt}\". Do you want help with uploads, history, forwarding or export?"
        else:
            # fallback short intelligent echo + prompt suggestions
            try:
                sents = sent_tokenize(prompt)
                snippet = (sents[0] if sents else prompt)[:200]
            except Exception:
                snippet = prompt[:200]
            reply_text = f"(Local assistant) I got: \"{snippet}\". Try: 'How many invoices?', 'Show recent uploads', 'Forward invoices', or 'Export invoices CSV'."

    # Persist assistant reply
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO chats (session_id, role, message, created_at) VALUES (?, ?, ?, ?)',
                  (session_id, 'assistant', reply_text, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        print("DB write (assistant) failed:", e)

    # Return reply
    return jsonify({'reply': reply_text})

@app.route('/chat/history')
@login_required
def chat_history():
    session_id = request.args.get('session_id', 'default')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT role, message FROM chats WHERE session_id = ? ORDER BY id ASC', (session_id,))
    rows = c.fetchall()
    conn.close()
    return jsonify({'history': [{'role': r, 'message': m} for r, m in rows]})

# ---------------------- History page + export ----------------------
@app.route('/history')
@login_required
def history_page():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    q = request.args.get('q', '').strip()
    category = request.args.get('category', '').strip()

    where_clauses = []
    params = []

    if category:
        where_clauses.append("category = ?")
        params.append(category)

    if q:
        where_clauses.append("(filename LIKE ? OR summary LIKE ? OR uploader_email LIKE ?)")
        like_q = f"%{q}%"
        params.extend([like_q, like_q, like_q])

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    count_sql = f"SELECT COUNT(*) FROM uploads {where_sql}"
    c.execute(count_sql, params)
    total = c.fetchone()[0]

    offset = (page - 1) * per_page
    select_sql = f"""
        SELECT id, filename, saved_path, summary, category, uploader_email, uploaded_at
        FROM uploads
        {where_sql}
        ORDER BY uploaded_at DESC
        LIMIT ? OFFSET ?
    """
    c.execute(select_sql, params + [per_page, offset])
    rows = c.fetchall()
    conn.close()

    total_pages = max(1, ceil(total / per_page))
    return render_template('history.html', rows=rows, page=page, per_page=per_page,
                           total=total, total_pages=total_pages, q=q, category=category)


@app.route('/history/export')
@login_required
def history_export():
    q = request.args.get('q', '').strip()
    category = request.args.get('category', '').strip()

    where_clauses = []
    params = []
    if category:
        where_clauses.append("category = ?")
        params.append(category)
    if q:
        where_clauses.append("(filename LIKE ? OR summary LIKE ? OR uploader_email LIKE ?)")
        like_q = f"%{q}%"
        params.extend([like_q, like_q, like_q])
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"SELECT id, filename, saved_path, summary, category, uploader_email, uploaded_at FROM uploads {where_sql} ORDER BY uploaded_at DESC", params)
    rows = c.fetchall()
    conn.close()

    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(['id','filename','saved_path','category','uploader_email','uploaded_at','summary'])
    for r in rows:
        writer.writerow([r[0], r[1], r[2] or '', r[4] or '', r[5] or '', r[6] or '', r[3] or ''])
    output = si.getvalue()
    si.close()
    resp = Response(output, mimetype='text/csv')
    resp.headers['Content-Disposition'] = 'attachment; filename=uploads_history.csv'
    return resp

# ---------------- Secure file serving ----------------
@app.route('/uploads/<path:filename>')
@login_required
def download_file(filename):
    uploads_dir = os.path.abspath(app.config.get('UPLOAD_FOLDER', UPLOAD_FOLDER))
    basename = filename.replace('\\', '/').split('/')[-1]

    candidate = os.path.join(uploads_dir, basename)
    if os.path.exists(candidate) and os.path.isfile(candidate):
        return send_from_directory(uploads_dir, basename, as_attachment=True)

    try:
        for root, _, files in os.walk(uploads_dir):
            for f in files:
                if f == basename or f.endswith(basename):
                    full_path = os.path.abspath(os.path.join(root, f))
                    if not full_path.startswith(uploads_dir):
                        continue
                    rel_path = os.path.relpath(full_path, uploads_dir)
                    rel_path_posix = rel_path.replace(os.path.sep, '/')
                    return send_from_directory(uploads_dir, rel_path_posix, as_attachment=True)
    except Exception as e:
        print("download_file search error:", e)

    return abort(404)

# ---------------------------------
# AUTH ROUTES (login/logout)
# ---------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    next_url = request.args.get('next') or url_for('index')
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username == ADMIN_USER and check_password_hash(_ADMIN_PASS_HASH, password):
            session['logged_in'] = True
            session['username'] = username
            flash("Login successful.", "success")
            return redirect(next_url)
        else:
            flash("Invalid credentials.", "error")
    return render_template('login.html', next=next_url)

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for('login'))


@app.route('/admin/settings', methods=['GET', 'POST'])
@admin_required
def admin_settings():
    runtime = get_runtime_settings()
    display = {
        'EMAIL_USER': runtime.get('email_user') or '',
        'SMTP_SERVER': runtime.get('smtp_server') or '',
        'SMTP_PORT': str(runtime.get('smtp_port') or 587),
        'IMAP_HOST': runtime.get('imap_host') or '',
        'IMAP_PORT': str(runtime.get('imap_port') or 993),
        'IMAP_USER': runtime.get('imap_user') or '',
        'ROUTE_invoice': runtime.get('route_invoice') or '',
        'ROUTE_payslip': runtime.get('route_payslip') or '',
        'ROUTE_purchase_order': runtime.get('route_purchase_order') or '',
        'ROUTE_minutes': runtime.get('route_minutes') or '',
        'FROM_NAME': runtime.get('from_name') or 'Smart Document Hub',
        'ADMIN_EMAIL': runtime.get('admin_email') or '',
    }
    has_email_pass = bool(runtime.get('email_pass'))
    has_imap_pass = bool(runtime.get('imap_pass'))

    if request.method == 'POST':
        form = request.form

        smtp_port = _safe_int(form.get('SMTP_PORT', '').strip(), -1)
        imap_port = _safe_int(form.get('IMAP_PORT', '').strip(), -1)

        if not (1 <= smtp_port <= 65535):
            flash('SMTP port must be between 1 and 65535.', 'error')
            return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass)
        if not (1 <= imap_port <= 65535):
            flash('IMAP port must be between 1 and 65535.', 'error')
            return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass)

        email_user = (form.get('EMAIL_USER') or '').strip()
        imap_user = (form.get('IMAP_USER') or '').strip()
        route_invoice = (form.get('ROUTE_invoice') or '').strip()
        route_payslip = (form.get('ROUTE_payslip') or '').strip()
        route_po = (form.get('ROUTE_purchase_order') or '').strip()
        route_minutes = (form.get('ROUTE_minutes') or '').strip()
        admin_email = (form.get('ADMIN_EMAIL') or '').strip()

        email_fields = [
            ('EMAIL_USER', email_user),
            ('IMAP_USER', imap_user),
            ('ROUTE_invoice', route_invoice),
            ('ROUTE_payslip', route_payslip),
            ('ROUTE_purchase_order', route_po),
            ('ROUTE_minutes', route_minutes),
        ]
        if admin_email:
            email_fields.append(('ADMIN_EMAIL', admin_email))

        for field_name, value in email_fields:
            if value and not _is_valid_email(value):
                flash(f'Invalid email format for {field_name}.', 'error')
                return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass)

        updates = {
            'email_user': email_user,
            'smtp_server': (form.get('SMTP_SERVER') or '').strip(),
            'smtp_port': str(smtp_port),
            'imap_host': (form.get('IMAP_HOST') or '').strip(),
            'imap_port': str(imap_port),
            'imap_user': imap_user,
            'route_invoice': route_invoice,
            'route_payslip': route_payslip,
            'route_purchase_order': route_po,
            'route_minutes': route_minutes,
            'from_name': (form.get('FROM_NAME') or '').strip() or 'Smart Document Hub',
            'admin_email': admin_email,
        }

        email_pass = (form.get('EMAIL_PASS') or '').strip()
        if email_pass:
            updates['email_pass'] = email_pass
        imap_pass = (form.get('IMAP_PASS') or '').strip()
        if imap_pass:
            updates['imap_pass'] = imap_pass

        try:
            for key, value in updates.items():
                set_setting(DB_PATH, key, value, encrypt=key in SENSITIVE_SETTING_KEYS)
        except Exception as e:
            flash(f'Failed to save settings: {e}', 'error')
            return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass)

        audit_log('settings_updated', 'Updated SMTP/IMAP credentials and routing settings.')

        # Apply changes immediately for IMAP worker without restart.
        try:
            from imap_fetcher import reload_runtime_config
            reload_runtime_config()
        except Exception as e:
            print('IMAP runtime reload failed:', e)

        flash('Settings updated successfully.', 'success')
        return redirect(url_for('admin_settings'))

    return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass)

# ---------------- IMAP monitoring helpers (non-invasive) ----------------
# We set a global thread reference when starting IMAP to allow status checks.
imap_thread = None

# Expose a small status endpoint to confirm IMAP thread is alive and env routes
@app.route('/imap_status')
@login_required
def imap_status():
    runtime = get_runtime_settings()
    info = {
        'imap_thread_alive': bool(imap_thread and imap_thread.is_alive()),
        'imap_thread_name': imap_thread.name if imap_thread else None,
        'smtp_user': runtime.get('email_user'),
        'settings_updated_at': get_settings_updated_at(DB_PATH),
        'routes': {
            'invoice': runtime.get('route_invoice'),
            'payslip': runtime.get('route_payslip'),
            'purchase_order': runtime.get('route_purchase_order'),
            'minutes': runtime.get('route_minutes')
        }
    }
    return jsonify(info)

@app.route('/status')
@login_required
def status():
    runtime = get_runtime_settings()
    return jsonify({
        'app': 'Smart Document Hub',
        'ml_model_loaded': bool(_ml_pipeline),
        'smtp_user': bool(runtime.get('email_user')),
        'imap_thread_alive': bool(imap_thread and imap_thread.is_alive()),
        'settings_updated_at': get_settings_updated_at(DB_PATH),
    })

# ---------------------------------
# MAIN
# ---------------------------------
if __name__ == '__main__':
    init_db()
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Start IMAP fetcher thread when running app directly if imap_fetcher exists
    try:
        from imap_fetcher import poll_imap_loop
        def start_imap_thread():
            global imap_thread
            t = threading.Thread(target=poll_imap_loop, name="imap-fetcher-thread", daemon=True)
            t.start()
            imap_thread = t
            print(">>> IMAP fetcher thread started (daemon).")
        start_imap_thread()
    except Exception as _e:
        print("IMAP fetcher not started:", _e)

    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', '5000'))
    debug = os.getenv('FLASK_DEBUG', '0') == '1'
    app.run(host=host, port=port, debug=debug)
