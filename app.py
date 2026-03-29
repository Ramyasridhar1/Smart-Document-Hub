# ---------------------- IMPORTS ----------------------
import os
import db_compat as sqlite3
import time
import threading
import shutil
import logging
from logging.handlers import RotatingFileHandler
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
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://smartdoc:smartdoc@localhost:5432/smartdoc')
# Backward-compatible alias used by existing helper calls.
DB_PATH = DATABASE_URL
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc'}

EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))

ROUTE_invoice = os.getenv('ROUTE_invoice')
ROUTE_payslip = os.getenv('ROUTE_payslip')
ROUTE_purchase_order = os.getenv('ROUTE_purchase_order')
ROUTE_minutes = os.getenv('ROUTE_minutes')
ROUTE_resume = os.getenv('ROUTE_resume')

FROM_NAME = os.getenv('FROM_NAME', 'Smart Document Hub')
ADMIN_EMAIL = os.getenv('ADMIN_EMAIL')

LOG_FILE_PATH = os.getenv('LOG_FILE_PATH', os.path.join('logs', 'smart_document_hub.log'))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

DEFAULT_ROUTE_DIRS = {
    'invoice': os.path.join('routed', 'invoice'),
    'payslip': os.path.join('routed', 'payslip'),
    'purchase_order': os.path.join('routed', 'purchase_order'),
    'minutes': os.path.join('routed', 'minutes'),
}

# ----------------- AUTH CONFIG -----------------
ADMIN_USER = os.getenv('ADMIN_USER', 'admin')
ADMIN_PASS = os.getenv('ADMIN_PASS', None)
# -----------------------------------------------

logger = logging.getLogger('smart_document_hub')


def _bool_from_str(value, default=True):
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def setup_logging(log_file_path=None, log_level=None):
    level_name = (log_level or LOG_LEVEL or 'INFO').upper()
    level = getattr(logging, level_name, logging.INFO)
    target_path = log_file_path or LOG_FILE_PATH
    target_path = os.path.abspath(target_path)

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)

    file_handler = RotatingFileHandler(target_path, maxBytes=2 * 1024 * 1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    logger.info('Logging configured. file=%s level=%s', target_path, level_name)

# ---------------------------------
# FLASK APP INIT
# ---------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATABASE_URL'] = DATABASE_URL
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB

SECRET_KEY = os.getenv('FLASK_SECRET_KEY', None) or os.getenv('SECRET_KEY', None) or os.urandom(24).hex()
app.secret_key = SECRET_KEY


def get_db_conn(path=None):
    return sqlite3.connect(path or DATABASE_URL)


def fetch_user_by_username(username):
    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute(
            '''SELECT id, username, password_hash, is_admin, is_active, created_at, updated_at, last_login_at
               FROM users WHERE username = ?''',
            (username,),
        )
        row = c.fetchone()
    finally:
        conn.close()

    if not row:
        return None
    return {
        'id': row[0],
        'username': row[1],
        'password_hash': row[2],
        'is_admin': bool(row[3]),
        'is_active': bool(row[4]),
        'created_at': row[5],
        'updated_at': row[6],
        'last_login_at': row[7],
    }


def fetch_user_by_id(user_id):
    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute(
            '''SELECT id, username, password_hash, is_admin, is_active, created_at, updated_at, last_login_at
               FROM users WHERE id = ?''',
            (user_id,),
        )
        row = c.fetchone()
    finally:
        conn.close()

    if not row:
        return None
    return {
        'id': row[0],
        'username': row[1],
        'password_hash': row[2],
        'is_admin': bool(row[3]),
        'is_active': bool(row[4]),
        'created_at': row[5],
        'updated_at': row[6],
        'last_login_at': row[7],
    }


def get_current_user():
    user_id = session.get('user_id')
    if not user_id:
        return None
    return fetch_user_by_id(user_id)


def is_admin_user():
    user = get_current_user()
    return bool(user and user.get('is_admin'))


# expose a few helpful vars to all templates so templates can use them directly
@app.context_processor
def inject_template_globals():
    runtime = get_runtime_settings()
    current_user = get_current_user()
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
        'current_user': current_user,
        'is_admin': bool(current_user and current_user.get('is_admin')),
    }

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user or not user.get('is_active'):
            session.clear()
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user or not user.get('is_active'):
            session.clear()
            return redirect(url_for('login', next=request.path))
        if not user.get('is_admin'):
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
    default_upload = app.config.get('UPLOAD_FOLDER', UPLOAD_FOLDER)
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
        'route_resume': ROUTE_resume,
        'from_name': FROM_NAME,
        'admin_email': ADMIN_EMAIL,
        'imap_poll_seconds': os.getenv('IMAP_POLL_SECONDS', '20'),
        'upload_folder': default_upload,
        'route_dir_invoice': os.path.join(default_upload, DEFAULT_ROUTE_DIRS['invoice']),
        'route_dir_payslip': os.path.join(default_upload, DEFAULT_ROUTE_DIRS['payslip']),
        'route_dir_purchase_order': os.path.join(default_upload, DEFAULT_ROUTE_DIRS['purchase_order']),
        'route_dir_minutes': os.path.join(default_upload, DEFAULT_ROUTE_DIRS['minutes']),
        'route_dir_resume': os.path.join(default_upload, 'routed', 'resume'),
        'route_local_enabled': os.getenv('ROUTE_LOCAL_ENABLED', '1'),
        'route_email_enabled': os.getenv('ROUTE_EMAIL_ENABLED', '1'),
        'log_file_path': os.getenv('LOG_FILE_PATH', LOG_FILE_PATH),
        'log_level': os.getenv('LOG_LEVEL', LOG_LEVEL),
        'resume_required_skills': os.getenv('RESUME_REQUIRED_SKILLS', 'python'),
        'resume_preferred_skills': os.getenv('RESUME_PREFERRED_SKILLS', ''),
        'resume_certificate_bonus': os.getenv('RESUME_CERTIFICATE_BONUS', '10'),
        'resume_project_bonus': os.getenv('RESUME_PROJECT_BONUS', '10'),
    }

    try:
        db_values = get_settings_bulk(DATABASE_URL)
        for key in DEFAULT_SETTING_KEYS:
            if key in db_values and db_values.get(key) is not None:
                settings[key] = db_values.get(key)
    except Exception:
        pass

    settings['smtp_port'] = _safe_int(settings.get('smtp_port'), 587)
    settings['imap_port'] = _safe_int(settings.get('imap_port'), 993)
    settings['imap_poll_seconds'] = _safe_int(settings.get('imap_poll_seconds'), 20)
    settings['route_local_enabled'] = _bool_from_str(settings.get('route_local_enabled'), default=True)
    settings['route_email_enabled'] = _bool_from_str(settings.get('route_email_enabled'), default=True)
    return settings


def audit_log(action, details=''):
    user = get_current_user()
    actor = user.get('username') if user else 'system'
    conn = sqlite3.connect(DATABASE_URL)
    try:
        c = conn.cursor()
        c.execute(
            'INSERT INTO audit_log (actor, action, details, created_at) VALUES (?, ?, ?, ?)',
            (actor, action, details, datetime.utcnow().isoformat()),
        )
        conn.commit()
    except Exception as e:
        logger.warning('Audit log failed: %s', e)
    finally:
        conn.close()

# ---------------------------------
# DATABASE INIT
# ---------------------------------
def init_db(db_url=None):
    global DATABASE_URL, DB_PATH
    resolved_db_url = db_url or DATABASE_URL
    DATABASE_URL = resolved_db_url
    DB_PATH = resolved_db_url
    app.config['DATABASE_URL'] = resolved_db_url

    conn = sqlite3.connect(resolved_db_url)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS uploads (
                    id BIGSERIAL PRIMARY KEY,
                    filename TEXT,
                    saved_path TEXT,
                    summary TEXT,
                    category TEXT,
                    uploader_email TEXT,
                    uploaded_at TEXT,
                    resume_score DOUBLE PRECISION,
                    resume_rank_note TEXT,
                    resume_risk_score DOUBLE PRECISION,
                    resume_risk_flags TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS chats (
                    id BIGSERIAL PRIMARY KEY,
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
                    id BIGSERIAL PRIMARY KEY,
                    actor TEXT,
                    action TEXT,
                    details TEXT,
                    created_at TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_admin INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT,
                    updated_at TEXT,
                    last_login_at TEXT
                )''')
    conn.commit()

    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS saved_path TEXT")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS resume_score DOUBLE PRECISION")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS resume_rank_note TEXT")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS resume_risk_score DOUBLE PRECISION")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS resume_risk_flags TEXT")
    conn.commit()

    # Bootstrap admin user if not present.
    admin_password = ADMIN_PASS or 'changeme'
    now = datetime.utcnow().isoformat()
    c.execute('SELECT id FROM users WHERE username = ?', (ADMIN_USER,))
    existing_admin = c.fetchone()
    if not existing_admin:
        c.execute(
            '''INSERT INTO users (username, password_hash, is_admin, is_active, created_at, updated_at)
               VALUES (?, ?, 1, 1, ?, ?)''',
            (ADMIN_USER, generate_password_hash(admin_password), now, now),
        )
        logger.info('Bootstrapped admin user: %s', ADMIN_USER)

    # Seed default path and control settings if missing.
    defaults = {
        'upload_folder': app.config.get('UPLOAD_FOLDER', UPLOAD_FOLDER),
        'route_dir_invoice': os.path.join(app.config.get('UPLOAD_FOLDER', UPLOAD_FOLDER), DEFAULT_ROUTE_DIRS['invoice']),
        'route_dir_payslip': os.path.join(app.config.get('UPLOAD_FOLDER', UPLOAD_FOLDER), DEFAULT_ROUTE_DIRS['payslip']),
        'route_dir_purchase_order': os.path.join(app.config.get('UPLOAD_FOLDER', UPLOAD_FOLDER), DEFAULT_ROUTE_DIRS['purchase_order']),
        'route_dir_minutes': os.path.join(app.config.get('UPLOAD_FOLDER', UPLOAD_FOLDER), DEFAULT_ROUTE_DIRS['minutes']),
        'route_dir_resume': os.path.join(app.config.get('UPLOAD_FOLDER', UPLOAD_FOLDER), 'routed', 'resume'),
        'route_local_enabled': '1',
        'route_email_enabled': '1',
        'log_file_path': LOG_FILE_PATH,
        'log_level': LOG_LEVEL,
        'route_resume': ROUTE_resume or '',
        'resume_required_skills': os.getenv('RESUME_REQUIRED_SKILLS', 'python'),
        'resume_preferred_skills': os.getenv('RESUME_PREFERRED_SKILLS', ''),
        'resume_certificate_bonus': os.getenv('RESUME_CERTIFICATE_BONUS', '10'),
        'resume_project_bonus': os.getenv('RESUME_PROJECT_BONUS', '10'),
    }
    for key, value in defaults.items():
        c.execute('SELECT 1 FROM settings WHERE key = ?', (key,))
        if not c.fetchone():
            c.execute(
                'INSERT INTO settings (key, value, is_encrypted, updated_at) VALUES (?, ?, 0, ?)',
                (key, value, now),
            )

    conn.commit()
    conn.close()
    ensure_settings_schema(resolved_db_url)

# ---------------------------------
# HELPERS
# ---------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _normalize_dir(path_value):
    cleaned = (path_value or '').strip()
    if not cleaned:
        return None
    return os.path.abspath(cleaned)


def get_route_output_dir(category, runtime):
    key_map = {
        'invoice': 'route_dir_invoice',
        'payslip': 'route_dir_payslip',
        'purchase_order': 'route_dir_purchase_order',
        'minutes': 'route_dir_minutes',
        'resume': 'route_dir_resume',
    }
    key = key_map.get(category)
    if not key:
        return None
    return _normalize_dir(runtime.get(key))


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
    if _ml_pipeline is not None:
        try:
            pred = _ml_pipeline.predict([text or ""])[0]
            if hasattr(_ml_pipeline, "predict_proba"):
                probs = _ml_pipeline.predict_proba([text or ""])[0]
                if max(probs) >= 0.45:
                    return str(pred)
            else:
                return str(pred)
        except Exception as e:
            logger.warning("ML classify fallback to keywords: %s", e)

    invoice_keywords = ['invoice', 'amount due', 'invoice no', 'bill to', 'total', 'tax', 'gst']
    payslip_keywords = ['payslip', 'salary', 'net pay', 'pay period', 'gross pay']
    po_keywords = ['purchase order', 'po no']
    minutes_keywords = ['minutes of meeting', 'attendees', 'agenda', 'meeting']
    resume_keywords = ['resume', 'curriculum vitae', 'experience', 'skills', 'education', 'certification', 'projects']

    scores = {
        'invoice': sum(t.count(k) for k in invoice_keywords),
        'payslip': sum(t.count(k) for k in payslip_keywords),
        'purchase_order': sum(t.count(k) for k in po_keywords),
        'minutes': sum(t.count(k) for k in minutes_keywords),
        'resume': sum(t.count(k) for k in resume_keywords),
    }

    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else 'other'


def parse_resume_preferences(runtime):
    required = [s.strip().lower() for s in (runtime.get('resume_required_skills') or '').split(',') if s.strip()]
    preferred = [s.strip().lower() for s in (runtime.get('resume_preferred_skills') or '').split(',') if s.strip()]
    cert_bonus = _safe_int(runtime.get('resume_certificate_bonus'), 10)
    project_bonus = _safe_int(runtime.get('resume_project_bonus'), 10)
    return {
        'required_skills': required,
        'preferred_skills': preferred,
        'certificate_bonus': cert_bonus,
        'project_bonus': project_bonus,
    }


def score_resume(text, prefs):
    t = (text or '').lower()
    score = 0.0
    notes = []

    for skill in prefs.get('required_skills', []):
        if skill in t:
            score += 25
            notes.append(f"required:{skill}")
    for skill in prefs.get('preferred_skills', []):
        if skill in t:
            score += 10
            notes.append(f"preferred:{skill}")

    if any(k in t for k in ['certificate', 'certified', 'certification']):
        score += max(0, prefs.get('certificate_bonus', 0))
        notes.append('certificate_bonus')
    if any(k in t for k in ['project', 'github', 'portfolio', 'implemented']):
        score += max(0, prefs.get('project_bonus', 0))
        notes.append('project_bonus')

    return round(score, 2), ', '.join(notes) if notes else 'No preference matches detected'


def assess_resume_risk(text):
    t = (text or '').lower()
    risk = 0.0
    flags = []

    if len((text or '').strip()) < 300:
        risk += 25
        flags.append('very_short_resume')
    if '@' not in t:
        risk += 15
        flags.append('missing_email')
    if not any(ch.isdigit() for ch in (text or '')):
        risk += 10
        flags.append('missing_dates_or_numbers')

    suspicious_phrases = ['100% guaranteed', 'world best', 'top 1%', 'expert in everything', 'perfect candidate']
    for phrase in suspicious_phrases:
        if phrase in t:
            risk += 12
            flags.append(f'suspicious_claim:{phrase}')

    python_hits = t.count('python')
    if python_hits > 20:
        risk += 20
        flags.append('possible_keyword_stuffing_python')

    return min(100.0, round(risk, 2)), flags


def log_upload(filename, saved_path, summary, category, uploader_email=None, resume_score=None, resume_rank_note=None, resume_risk_score=None, resume_risk_flags=None):
    conn = sqlite3.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute(
        'INSERT INTO uploads (filename, saved_path, summary, category, uploader_email, uploaded_at, resume_score, resume_rank_note, resume_risk_score, resume_risk_flags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (
            filename,
            saved_path,
            summary,
            category,
            uploader_email,
            datetime.utcnow().isoformat(),
            resume_score,
            resume_rank_note,
            resume_risk_score,
            ', '.join(resume_risk_flags or []) if isinstance(resume_risk_flags, list) else resume_risk_flags,
        )
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
        'minutes': runtime.get('route_minutes'),
        'resume': runtime.get('route_resume'),
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
    runtime = get_runtime_settings()
    upload_root = _normalize_dir(runtime.get('upload_folder')) or os.path.abspath(UPLOAD_FOLDER)
    app.config['UPLOAD_FOLDER'] = upload_root

    for file in files:
        if not (file and allowed_file(file.filename)):
            continue

        filename = secure_filename(file.filename)
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        saved_path = os.path.join(upload_root, saved_filename)

        os.makedirs(upload_root, exist_ok=True)
        file.save(saved_path)

        # Extract, summarize, classify
        text = extract_text(saved_path)
        summary = simple_summarize(text) if text else "No text extracted."
        category = classify_document(text)
        resume_score = None
        resume_rank_note = None
        resume_risk_score = None
        resume_risk_flags = None

        if category == 'resume':
            prefs = parse_resume_preferences(runtime)
            resume_score, resume_rank_note = score_resume(text, prefs)
            resume_risk_score, resume_risk_flags = assess_resume_risk(text)

        # Create category folder and move file there
        cat_folder = (category if category else 'other')
        cat_folder = "".join(ch for ch in cat_folder if ch.isalnum() or ch in ('_', '-')).lower() or 'other'
        target_dir = os.path.join(upload_root, cat_folder)
        os.makedirs(target_dir, exist_ok=True)
        dest_path = os.path.join(target_dir, saved_filename)
        try:
            shutil.move(saved_path, dest_path)
            saved_path = dest_path
        except Exception as e:
            print("⚠️ Warning: failed to move file to category folder:", e)
            # saved_path remains original if move fails

        # Log upload
        log_upload(
            saved_filename,
            saved_path,
            summary,
            category,
            uploader_email,
            resume_score=resume_score,
            resume_rank_note=resume_rank_note,
            resume_risk_score=resume_risk_score,
            resume_risk_flags=resume_risk_flags,
        )

        # Optional local routing output copy.
        if runtime.get('route_local_enabled'):
            route_dir = get_route_output_dir(category, runtime)
            if route_dir:
                try:
                    os.makedirs(route_dir, exist_ok=True)
                    routed_path = os.path.join(route_dir, saved_filename)
                    shutil.copy2(saved_path, routed_path)
                except Exception as e:
                    logger.warning('Failed to copy routed file to %s: %s', route_dir, e)

        # NOTE: per-file forwarding removed here (we do batch forwarding after the loop)

        # record result details for later (including route target if configured)
        forward_to = route_for_category(category)
        results.append({
            'filename': saved_filename,
            'category': category,
            'summary': summary,
            'route': forward_to,
            'saved_path': saved_path,
            'uploader_email': uploader_email,
            'resume_score': resume_score,
            'resume_rank_note': resume_rank_note,
            'resume_risk_score': resume_risk_score,
            'resume_risk_flags': resume_risk_flags,
        })

    if not results:
        return "No valid files uploaded.", 400

    # ------------------ BATCH FORWARDING (group by recipient) ------------------
    from collections import defaultdict
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    batch_map = defaultdict(list)

    # Group items by recipient (route field)
    if runtime.get('route_email_enabled'):
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
        csv_path = os.path.join(upload_root, csv_name)
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
        SELECT id, filename, saved_path, summary, category, uploader_email, uploaded_at, resume_score, resume_risk_score, resume_risk_flags
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
    c.execute(f"SELECT id, filename, saved_path, summary, category, uploader_email, uploaded_at, resume_score, resume_risk_score, resume_risk_flags FROM uploads {where_sql} ORDER BY uploaded_at DESC", params)
    rows = c.fetchall()
    conn.close()

    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(['id','filename','saved_path','category','uploader_email','uploaded_at','summary','resume_score','resume_risk_score','resume_risk_flags'])
    for r in rows:
        writer.writerow([r[0], r[1], r[2] or '', r[4] or '', r[5] or '', r[6] or '', r[3] or '', r[7] or '', r[8] or '', r[9] or ''])
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
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = fetch_user_by_username(username)
        if not user:
            flash("Invalid credentials.", "error")
            return render_template('login.html', next=next_url)

        if not user.get('is_active'):
            flash("Your account is inactive. Contact an administrator.", "error")
            return render_template('login.html', next=next_url)

        if not check_password_hash(user.get('password_hash'), password):
            flash("Invalid credentials.", "error")
            return render_template('login.html', next=next_url)

        conn = get_db_conn()
        try:
            c = conn.cursor()
            c.execute('UPDATE users SET last_login_at = ?, updated_at = ? WHERE id = ?',
                      (datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), user['id']))
            conn.commit()
        finally:
            conn.close()

        session['user_id'] = user['id']
        session['username'] = user['username']
        session['is_admin'] = bool(user.get('is_admin'))

        audit_log('user_login', f"User {user['username']} logged in")
        flash("Login successful.", "success")
        return redirect(next_url)
    return render_template('login.html', next=next_url)

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for('login'))


@app.route('/account/password', methods=['POST'])
@login_required
def change_password():
    user = get_current_user()
    current_password = (request.form.get('current_password') or '').strip()
    new_password = (request.form.get('new_password') or '').strip()

    if not check_password_hash(user['password_hash'], current_password):
        flash('Current password is incorrect.', 'error')
        return redirect(request.referrer or url_for('index'))
    if len(new_password) < 8:
        flash('New password must be at least 8 characters.', 'error')
        return redirect(request.referrer or url_for('index'))

    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute(
            'UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?',
            (generate_password_hash(new_password), datetime.utcnow().isoformat(), user['id']),
        )
        conn.commit()
    finally:
        conn.close()

    audit_log('password_changed', f"User {user['username']} changed own password")
    flash('Password updated.', 'success')
    return redirect(request.referrer or url_for('index'))


@app.route('/admin')
@admin_required
def admin_dashboard():
    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM users')
        total_users = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM users WHERE is_admin = 1')
        total_admins = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
        active_users = c.fetchone()[0]
    finally:
        conn.close()

    runtime = get_runtime_settings()
    return render_template(
        'admin.html',
        total_users=total_users,
        total_admins=total_admins,
        active_users=active_users,
        runtime=runtime,
    )


@app.route('/admin/users', methods=['GET', 'POST'])
@admin_required
def admin_users():
    current_user = get_current_user()

    if request.method == 'POST':
        action = (request.form.get('action') or '').strip()
        target_id = _safe_int(request.form.get('user_id', '').strip(), -1)

        conn = get_db_conn()
        try:
            c = conn.cursor()
            if action == 'create_user':
                username = (request.form.get('username') or '').strip()
                password = (request.form.get('password') or '').strip()
                is_admin_new = 1 if request.form.get('is_admin') == 'on' else 0
                if len(username) < 3 or len(password) < 8:
                    flash('Username must be at least 3 chars and password at least 8 chars.', 'error')
                else:
                    c.execute('SELECT 1 FROM users WHERE username = ?', (username,))
                    if c.fetchone():
                        flash('Username already exists.', 'error')
                    else:
                        now = datetime.utcnow().isoformat()
                        c.execute(
                            '''INSERT INTO users (username, password_hash, is_admin, is_active, created_at, updated_at)
                               VALUES (?, ?, ?, 1, ?, ?)''',
                            (username, generate_password_hash(password), is_admin_new, now, now),
                        )
                        conn.commit()
                        audit_log('user_created', f"Created user={username}, is_admin={is_admin_new}")
                        flash('User created successfully.', 'success')

            elif action in {'toggle_admin', 'toggle_active', 'reset_password'} and target_id > 0:
                c.execute('SELECT id, username, is_admin, is_active FROM users WHERE id = ?', (target_id,))
                target = c.fetchone()
                if not target:
                    flash('User not found.', 'error')
                else:
                    uid, uname, is_admin_val, is_active_val = target
                    if uid == current_user['id'] and action in {'toggle_admin', 'toggle_active'}:
                        flash('You cannot change your own admin/active status from this screen.', 'error')
                    else:
                        if action == 'toggle_admin':
                            next_value = 0 if is_admin_val else 1
                            c.execute('UPDATE users SET is_admin = ?, updated_at = ? WHERE id = ?',
                                      (next_value, datetime.utcnow().isoformat(), uid))
                            conn.commit()
                            audit_log('user_role_changed', f"User={uname}, is_admin={next_value}")
                            flash('Admin role updated.', 'success')
                        elif action == 'toggle_active':
                            next_value = 0 if is_active_val else 1
                            c.execute('UPDATE users SET is_active = ?, updated_at = ? WHERE id = ?',
                                      (next_value, datetime.utcnow().isoformat(), uid))
                            conn.commit()
                            audit_log('user_status_changed', f"User={uname}, is_active={next_value}")
                            flash('User status updated.', 'success')
                        elif action == 'reset_password':
                            new_password = (request.form.get('new_password') or '').strip()
                            if len(new_password) < 8:
                                flash('Reset password must be at least 8 characters.', 'error')
                            else:
                                c.execute('UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?',
                                          (generate_password_hash(new_password), datetime.utcnow().isoformat(), uid))
                                conn.commit()
                                audit_log('user_password_reset', f"Password reset for user={uname}")
                                flash('Password reset successful.', 'success')
            else:
                flash('Invalid action.', 'error')
        except sqlite3.IntegrityError:
            flash('Operation failed due to duplicate or invalid values.', 'error')
        finally:
            conn.close()

        return redirect(url_for('admin_users'))

    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute(
            '''SELECT id, username, is_admin, is_active, created_at, last_login_at
               FROM users ORDER BY username ASC'''
        )
        users = c.fetchall()
    finally:
        conn.close()

    return render_template('admin_users.html', users=users)


@app.route('/admin/settings', methods=['GET', 'POST'])
@admin_required
def admin_settings():
    global LOG_FILE_PATH, LOG_LEVEL

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
        'ROUTE_resume': runtime.get('route_resume') or '',
        'FROM_NAME': runtime.get('from_name') or 'Smart Document Hub',
        'ADMIN_EMAIL': runtime.get('admin_email') or '',
        'UPLOAD_FOLDER': runtime.get('upload_folder') or app.config.get('UPLOAD_FOLDER', UPLOAD_FOLDER),
        'ROUTE_DIR_invoice': runtime.get('route_dir_invoice') or '',
        'ROUTE_DIR_payslip': runtime.get('route_dir_payslip') or '',
        'ROUTE_DIR_purchase_order': runtime.get('route_dir_purchase_order') or '',
        'ROUTE_DIR_minutes': runtime.get('route_dir_minutes') or '',
        'ROUTE_DIR_resume': runtime.get('route_dir_resume') or '',
        'ROUTE_LOCAL_ENABLED': '1' if runtime.get('route_local_enabled') else '0',
        'ROUTE_EMAIL_ENABLED': '1' if runtime.get('route_email_enabled') else '0',
        'LOG_FILE_PATH': runtime.get('log_file_path') or LOG_FILE_PATH,
        'LOG_LEVEL': runtime.get('log_level') or LOG_LEVEL,
        'RESUME_REQUIRED_SKILLS': runtime.get('resume_required_skills') or '',
        'RESUME_PREFERRED_SKILLS': runtime.get('resume_preferred_skills') or '',
        'RESUME_CERTIFICATE_BONUS': str(runtime.get('resume_certificate_bonus') or '10'),
        'RESUME_PROJECT_BONUS': str(runtime.get('resume_project_bonus') or '10'),
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
        route_resume = (form.get('ROUTE_resume') or '').strip()
        admin_email = (form.get('ADMIN_EMAIL') or '').strip()

        email_fields = [
            ('EMAIL_USER', email_user),
            ('IMAP_USER', imap_user),
            ('ROUTE_invoice', route_invoice),
            ('ROUTE_payslip', route_payslip),
            ('ROUTE_purchase_order', route_po),
            ('ROUTE_minutes', route_minutes),
            ('ROUTE_resume', route_resume),
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
            'route_resume': route_resume,
            'from_name': (form.get('FROM_NAME') or '').strip() or 'Smart Document Hub',
            'admin_email': admin_email,
            'upload_folder': (form.get('UPLOAD_FOLDER') or '').strip(),
            'route_dir_invoice': (form.get('ROUTE_DIR_invoice') or '').strip(),
            'route_dir_payslip': (form.get('ROUTE_DIR_payslip') or '').strip(),
            'route_dir_purchase_order': (form.get('ROUTE_DIR_purchase_order') or '').strip(),
            'route_dir_minutes': (form.get('ROUTE_DIR_minutes') or '').strip(),
            'route_dir_resume': (form.get('ROUTE_DIR_resume') or '').strip(),
            'route_local_enabled': '1' if form.get('ROUTE_LOCAL_ENABLED') == 'on' else '0',
            'route_email_enabled': '1' if form.get('ROUTE_EMAIL_ENABLED') == 'on' else '0',
            'log_file_path': (form.get('LOG_FILE_PATH') or '').strip(),
            'log_level': (form.get('LOG_LEVEL') or '').strip().upper() or 'INFO',
            'resume_required_skills': (form.get('RESUME_REQUIRED_SKILLS') or '').strip(),
            'resume_preferred_skills': (form.get('RESUME_PREFERRED_SKILLS') or '').strip(),
            'resume_certificate_bonus': str(_safe_int(form.get('RESUME_CERTIFICATE_BONUS'), 10)),
            'resume_project_bonus': str(_safe_int(form.get('RESUME_PROJECT_BONUS'), 10)),
        }

        path_fields = [
            'upload_folder', 'route_dir_invoice', 'route_dir_payslip',
            'route_dir_purchase_order', 'route_dir_minutes', 'route_dir_resume', 'log_file_path'
        ]
        for key in path_fields:
            p = _normalize_dir(updates.get(key))
            if not p:
                flash(f'{key} is required.', 'error')
                return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass)
            updates[key] = p

        try:
            os.makedirs(updates['upload_folder'], exist_ok=True)
            os.makedirs(os.path.dirname(updates['log_file_path']), exist_ok=True)
            os.makedirs(updates['route_dir_invoice'], exist_ok=True)
            os.makedirs(updates['route_dir_payslip'], exist_ok=True)
            os.makedirs(updates['route_dir_purchase_order'], exist_ok=True)
            os.makedirs(updates['route_dir_minutes'], exist_ok=True)
            os.makedirs(updates['route_dir_resume'], exist_ok=True)
        except Exception as e:
            flash(f'Failed to create one or more configured directories: {e}', 'error')
            return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass)

        email_pass = (form.get('EMAIL_PASS') or '').strip()
        if email_pass:
            updates['email_pass'] = email_pass
        imap_pass = (form.get('IMAP_PASS') or '').strip()
        if imap_pass:
            updates['imap_pass'] = imap_pass

        try:
            for key, value in updates.items():
                set_setting(DATABASE_URL, key, value, encrypt=key in SENSITIVE_SETTING_KEYS)
        except Exception as e:
            flash(f'Failed to save settings: {e}', 'error')
            return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass)

        # Runtime apply for paths and logging.
        app.config['UPLOAD_FOLDER'] = updates['upload_folder']
        LOG_FILE_PATH = updates['log_file_path']
        LOG_LEVEL = updates['log_level']
        setup_logging(LOG_FILE_PATH, LOG_LEVEL)

        audit_log('settings_updated', 'Updated SMTP/IMAP, routes, resume ranking, storage, and logging settings.')

        # Apply changes immediately for IMAP worker without restart.
        try:
            from imap_fetcher import reload_runtime_config
            reload_runtime_config()
        except Exception as e:
            logger.warning('IMAP runtime reload failed: %s', e)

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
        'settings_updated_at': get_settings_updated_at(DATABASE_URL),
        'routes': {
            'invoice': runtime.get('route_invoice'),
            'payslip': runtime.get('route_payslip'),
            'purchase_order': runtime.get('route_purchase_order'),
            'minutes': runtime.get('route_minutes'),
            'resume': runtime.get('route_resume'),
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
        'settings_updated_at': get_settings_updated_at(DATABASE_URL),
        'upload_folder': runtime.get('upload_folder'),
        'database_backend': 'postgresql',
        'log_file_path': runtime.get('log_file_path'),
    })

# ---------------------------------
# MAIN
# ---------------------------------
if __name__ == '__main__':
    init_db()
    runtime = get_runtime_settings()

    app.config['UPLOAD_FOLDER'] = _normalize_dir(runtime.get('upload_folder')) or os.path.abspath(UPLOAD_FOLDER)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    setup_logging(runtime.get('log_file_path'), runtime.get('log_level'))

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
