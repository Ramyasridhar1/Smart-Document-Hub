# ---------------------- IMPORTS ----------------------
import os
import base64
import db_compat as sqlite3
import time
import threading
import shutil
import logging
from collections import deque
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
from PIL import Image, ImageOps, ImageFilter
import joblib
from math import ceil
import csv
from io import StringIO
from functools import wraps
from flask import session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import openai
import spacy
import re
from settings_store import (
    DEFAULT_SETTING_KEYS,
    SENSITIVE_SETTING_KEYS,
    ensure_settings_schema,
    get_settings_bulk,
    get_settings_updated_at,
    set_setting,
)


# ---------------------------------
# LOAD ENVIRONMENT
# ---------------------------------
load_dotenv()


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
CLASSIFIER_MODEL_PATH = os.getenv('CLASSIFIER_MODEL_PATH', os.path.join("model", "tfidf_logreg.joblib"))
RESUME_RANKER_MODEL_PATH = os.getenv('RESUME_RANKER_MODEL_PATH', os.path.join("model", "resume_ranker.joblib"))


def _load_joblib_model(path, label):
    if not os.path.exists(path):
        print(f">>> {label} model not found at", path)
        return None, 'unavailable'
    try:
        model = joblib.load(path)
        version = datetime.utcfromtimestamp(os.path.getmtime(path)).strftime('%Y%m%d%H%M%S')
        print(f">>> {label} model loaded from:", path)
        return model, version
    except Exception as exc:
        print(f">>> Failed to load {label} model:", exc)
        return None, 'unavailable'


_ml_pipeline, CLASSIFIER_MODEL_VERSION = _load_joblib_model(CLASSIFIER_MODEL_PATH, 'classifier')
_ml_resume_ranker, RESUME_RANKER_MODEL_VERSION = _load_joblib_model(RESUME_RANKER_MODEL_PATH, 'resume ranker')
# Backward-compatible alias used by existing DB/audit fields.
MODEL_VERSION = CLASSIFIER_MODEL_VERSION
# ----------------------------------------------------

# ---------------------------------
# APP CONFIG
# ---------------------------------
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://smartdoc:smartdoc@localhost:5432/smartdoc')
# Backward-compatible alias used by existing helper calls.
DB_PATH = DATABASE_URL
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc'}
CLASSIFICATION_CATEGORIES = ['invoice', 'payslip', 'purchase_order', 'minutes', 'resume', 'other']

# Document extraction & OCR settings
_enable_ocr_str = os.getenv('ENABLE_OCR', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
ENABLE_OCR = _enable_ocr_str  # OCR disabled by default (slow)
OCR_DPI = int(os.getenv('OCR_DPI', '200'))  # Lower DPI = faster OCR
EXTRACT_MAX_TEXT_BYTES = int(os.getenv('EXTRACT_MAX_TEXT_BYTES', '50000'))  # Limit text size
EXTRACT_PDF_MAX_PAGES = int(os.getenv('EXTRACT_PDF_MAX_PAGES', '10'))  # Only read first N pages
AUTO_ROUTE_CONFIDENCE_THRESHOLD = float(os.getenv('AUTO_ROUTE_CONFIDENCE_THRESHOLD', '0.75'))
INBOUND_ADAPTER = (os.getenv('INBOUND_ADAPTER', 'hybrid') or 'hybrid').strip().lower()
WEBHOOK_SHARED_SECRET = os.getenv('WEBHOOK_SHARED_SECRET', '')
ENABLE_RESUME_RANKER = str(os.getenv('ENABLE_RESUME_RANKER', '0')).strip().lower() in {'1', 'true', 'yes', 'on'}
try:
    RANKER_MIN_FIT_SCORE = float(os.getenv('RANKER_MIN_FIT_SCORE', '60'))
except Exception:
    RANKER_MIN_FIT_SCORE = 60.0
RANKER_MIN_FIT_SCORE = max(0.0, min(100.0, RANKER_MIN_FIT_SCORE))

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


def developer_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user or not user.get('is_active'):
            session.clear()
            return redirect(url_for('login', next=request.path))
        if not user.get('is_admin'):
            return abort(403)
        if (user.get('username') or '').strip() != (ADMIN_USER or '').strip():
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


def _safe_float(value, default):
    try:
        return float(str(value).strip())
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
        'imap_skip_existing_unseen': os.getenv('IMAP_SKIP_EXISTING_UNSEEN', '1'),
        'imap_allowed_senders': os.getenv('IMAP_ALLOWED_SENDERS', ''),
        'imap_subject_keyword': os.getenv('IMAP_SUBJECT_KEYWORD', ''),
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
        'auto_route_confidence_threshold': os.getenv('AUTO_ROUTE_CONFIDENCE_THRESHOLD', str(AUTO_ROUTE_CONFIDENCE_THRESHOLD)),
        'auto_route_threshold_invoice': os.getenv('AUTO_ROUTE_THRESHOLD_INVOICE', ''),
        'auto_route_threshold_payslip': os.getenv('AUTO_ROUTE_THRESHOLD_PAYSLIP', ''),
        'auto_route_threshold_purchase_order': os.getenv('AUTO_ROUTE_THRESHOLD_PURCHASE_ORDER', ''),
        'auto_route_threshold_minutes': os.getenv('AUTO_ROUTE_THRESHOLD_MINUTES', ''),
        'auto_route_threshold_resume': os.getenv('AUTO_ROUTE_THRESHOLD_RESUME', ''),
        'auto_route_threshold_other': os.getenv('AUTO_ROUTE_THRESHOLD_OTHER', ''),
        'inbound_adapter': os.getenv('INBOUND_ADAPTER', INBOUND_ADAPTER),
        'webhook_shared_secret': os.getenv('WEBHOOK_SHARED_SECRET', WEBHOOK_SHARED_SECRET),
        'enable_resume_ranker': os.getenv('ENABLE_RESUME_RANKER', '0'),
        'ranker_min_fit_score': os.getenv('RANKER_MIN_FIT_SCORE', '60'),
        'classifier_model_path': os.getenv('CLASSIFIER_MODEL_PATH', CLASSIFIER_MODEL_PATH),
        'resume_ranker_model_path': os.getenv('RESUME_RANKER_MODEL_PATH', RESUME_RANKER_MODEL_PATH),
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
    settings['auto_route_confidence_threshold'] = max(0.0, min(1.0, _safe_float(settings.get('auto_route_confidence_threshold'), AUTO_ROUTE_CONFIDENCE_THRESHOLD)))
    settings['route_local_enabled'] = _bool_from_str(settings.get('route_local_enabled'), default=True)
    settings['route_email_enabled'] = _bool_from_str(settings.get('route_email_enabled'), default=True)
    settings['inbound_adapter'] = (settings.get('inbound_adapter') or INBOUND_ADAPTER).strip().lower()
    settings['enable_resume_ranker'] = _bool_from_str(settings.get('enable_resume_ranker'), default=ENABLE_RESUME_RANKER)
    settings['ranker_min_fit_score'] = max(0.0, min(100.0, _safe_float(settings.get('ranker_min_fit_score'), RANKER_MIN_FIT_SCORE)))
    return settings


def get_auto_route_threshold(category, runtime):
    global_threshold = max(0.0, min(1.0, _safe_float(runtime.get('auto_route_confidence_threshold'), AUTO_ROUTE_CONFIDENCE_THRESHOLD)))
    specific_key = f'auto_route_threshold_{category}'
    raw = runtime.get(specific_key)
    if raw is None or str(raw).strip() == '':
        return global_threshold
    return max(0.0, min(1.0, _safe_float(raw, global_threshold)))


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
                    resume_risk_flags TEXT,
                    resume_fit_score DOUBLE PRECISION,
                    resume_fit_confidence DOUBLE PRECISION,
                    ml_confidence DOUBLE PRECISION,
                    top_candidates TEXT,
                    model_version TEXT,
                    classifier_version TEXT,
                    ranker_version TEXT,
                    processing_status TEXT,
                    processing_error TEXT
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
    c.execute('''CREATE TABLE IF NOT EXISTS review_feedback (
                    id BIGSERIAL PRIMARY KEY,
                    upload_id BIGINT,
                    admin_action TEXT,
                    from_category TEXT,
                    to_category TEXT,
                    shortlist_fit TEXT,
                    extraction_feedback TEXT,
                    reviewer_note TEXT,
                    actor TEXT,
                    created_at TEXT
                )''')
    conn.commit()

    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS saved_path TEXT")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS resume_score DOUBLE PRECISION")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS resume_rank_note TEXT")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS resume_risk_score DOUBLE PRECISION")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS resume_risk_flags TEXT")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS resume_fit_score DOUBLE PRECISION")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS resume_fit_confidence DOUBLE PRECISION")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS ml_confidence DOUBLE PRECISION")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS top_candidates TEXT")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS model_version TEXT")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS classifier_version TEXT")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS ranker_version TEXT")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS processing_status TEXT")
    c.execute("ALTER TABLE uploads ADD COLUMN IF NOT EXISTS processing_error TEXT")
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
        'auto_route_confidence_threshold': os.getenv('AUTO_ROUTE_CONFIDENCE_THRESHOLD', str(AUTO_ROUTE_CONFIDENCE_THRESHOLD)),
        'auto_route_threshold_invoice': os.getenv('AUTO_ROUTE_THRESHOLD_INVOICE', ''),
        'auto_route_threshold_payslip': os.getenv('AUTO_ROUTE_THRESHOLD_PAYSLIP', ''),
        'auto_route_threshold_purchase_order': os.getenv('AUTO_ROUTE_THRESHOLD_PURCHASE_ORDER', ''),
        'auto_route_threshold_minutes': os.getenv('AUTO_ROUTE_THRESHOLD_MINUTES', ''),
        'auto_route_threshold_resume': os.getenv('AUTO_ROUTE_THRESHOLD_RESUME', ''),
        'auto_route_threshold_other': os.getenv('AUTO_ROUTE_THRESHOLD_OTHER', ''),
        'inbound_adapter': os.getenv('INBOUND_ADAPTER', INBOUND_ADAPTER),
        'enable_resume_ranker': os.getenv('ENABLE_RESUME_RANKER', '0'),
        'ranker_min_fit_score': os.getenv('RANKER_MIN_FIT_SCORE', '60'),
        'classifier_model_path': os.getenv('CLASSIFIER_MODEL_PATH', CLASSIFIER_MODEL_PATH),
        'resume_ranker_model_path': os.getenv('RESUME_RANKER_MODEL_PATH', RESUME_RANKER_MODEL_PATH),
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


def _guess_quick_category_from_file(saved_path):
    try:
        with open(saved_path, 'rb') as f:
            peek_text = f.read(5000).decode('utf-8', errors='ignore').lower()
    except Exception:
        peek_text = ''

    if 'invoice' in peek_text or 'amount due' in peek_text:
        return 'invoice'
    if 'payslip' in peek_text or 'salary' in peek_text:
        return 'payslip'
    if 'purchase order' in peek_text or 'po no' in peek_text:
        return 'purchase_order'
    if 'minutes' in peek_text or 'agenda' in peek_text:
        return 'minutes'
    if 'resume' in peek_text or 'curriculum vitae' in peek_text:
        return 'resume'
    return 'other'


def _save_and_queue_file(saved_filename, source_path, upload_root, uploader_email):
    quick_category = _guess_quick_category_from_file(source_path)
    cat_folder = ''.join(ch for ch in quick_category if ch.isalnum() or ch in ('_', '-')).lower() or 'other'
    target_dir = os.path.join(upload_root, cat_folder)
    os.makedirs(target_dir, exist_ok=True)
    dest_path = os.path.join(target_dir, saved_filename)
    try:
        shutil.move(source_path, dest_path)
        source_path = dest_path
    except Exception as e:
        logger.warning('Move to category folder failed for %s: %s', saved_filename, e)

    upload_id = create_upload_pending(saved_filename, source_path, uploader_email)
    return {
        'filename': saved_filename,
        'saved_path': source_path,
        'uploader_email': uploader_email,
        'quick_category': quick_category,
        'upload_id': upload_id,
    }


def _normalize_webhook_json_payload(payload):
    """Normalize provider-specific webhook payloads to (sender, attachments)."""
    sender = None
    attachments = []
    provider = str(payload.get('provider') or payload.get('source') or '').strip().lower()

    # Sender detection across common payload styles.
    sender = (
        payload.get('sender')
        or payload.get('uploader_email')
        or payload.get('email')
        or payload.get('from')
    )
    if isinstance(sender, dict):
        sender = sender.get('emailAddress', {}).get('address') or sender.get('address')

    message = payload.get('message') if isinstance(payload.get('message'), dict) else {}
    if not sender:
        sender = message.get('from')
        if isinstance(sender, dict):
            sender = sender.get('emailAddress', {}).get('address') or sender.get('address')

    candidate_lists = []
    if isinstance(payload.get('attachments'), list):
        candidate_lists.append(payload.get('attachments'))
    if isinstance(message.get('attachments'), list):
        candidate_lists.append(message.get('attachments'))

    # Microsoft Graph notifications often carry resourceData/value arrays.
    if isinstance(payload.get('value'), list):
        for item in payload.get('value'):
            if not isinstance(item, dict):
                continue
            resource_data = item.get('resourceData') if isinstance(item.get('resourceData'), dict) else {}
            if isinstance(resource_data.get('attachments'), list):
                candidate_lists.append(resource_data.get('attachments'))
            if not sender:
                sender_candidate = item.get('from') or resource_data.get('from')
                if isinstance(sender_candidate, dict):
                    sender_candidate = sender_candidate.get('emailAddress', {}).get('address') or sender_candidate.get('address')
                sender = sender or sender_candidate

    for items in candidate_lists:
        for att in items:
            if not isinstance(att, dict):
                continue
            name = att.get('filename') or att.get('name') or att.get('fileName') or att.get('attachmentName')
            content_b64 = att.get('contentBytes') or att.get('data') or att.get('content') or att.get('base64')
            if not name or not content_b64:
                continue
            filename = secure_filename(str(name))
            if not filename or not allowed_file(filename):
                continue
            try:
                raw = base64.b64decode(str(content_b64), validate=False)
            except Exception:
                continue
            attachments.append({'filename': filename, 'content': raw})

    return {
        'provider': provider or 'generic-json',
        'sender': (sender or '').strip() or None,
        'attachments': attachments,
    }


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
    if not text:
        return "No text extracted."

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "No text extracted."

    # Prefer spaCy sentence boundaries when model is available.
    if nlp:
        try:
            doc = nlp(cleaned)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text and sent.text.strip()]
            if sentences:
                return ' '.join(sentences[:max_sentences])
        except Exception as e:
            logger.warning("spaCy summarization fallback triggered: %s", e)

    # Fallback for environments where spaCy model/sentencizer is unavailable.
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    parts = [p.strip() for p in parts if p and p.strip()]
    if parts:
        return ' '.join(parts[:max_sentences])

    return cleaned[:500]


def extract_text(file_path, ocr_dpi=None, max_pages_for_ocr=None, enable_ocr=None, timeout_seconds=10):
    """Extract text from files with OCR disabled by default (too slow).
    
    Args:
        file_path: Path to file to extract from
        ocr_dpi: DPI for OCR (lower=faster but lower quality, default from OCR_DPI env)
        max_pages_for_ocr: Max pages to OCR (default from env)
        enable_ocr: If False, skip OCR entirely. Default from ENABLE_OCR env var.
        timeout_seconds: Timeout for OCR operations
    """
    if not file_path or not os.path.exists(file_path):
        return ""
    
    # Use global config if not overridden
    if ocr_dpi is None:
        ocr_dpi = OCR_DPI
    if max_pages_for_ocr is None:
        max_pages_for_ocr = 3
    if enable_ocr is None:
        enable_ocr = ENABLE_OCR

    ext = file_path.rsplit('.', 1)[-1].lower()

    def _prepare_image_for_ocr(img):
        try:
            processed = ImageOps.exif_transpose(img)
            processed = processed.convert('L')
            # Upscale small images to improve OCR legibility on scanned docs.
            if processed.width < 1400:
                scale = max(1, int(1400 / max(1, processed.width)))
                if scale > 1:
                    processed = processed.resize((processed.width * scale, processed.height * scale), Image.Resampling.LANCZOS)
            processed = ImageOps.autocontrast(processed)
            processed = processed.filter(ImageFilter.MedianFilter(size=3))
            return processed
        except Exception:
            return img

    def _ocr_with_fallback(img, timeout=10):
        try:
            prepared = _prepare_image_for_ocr(img)
            binarized = prepared.point(lambda p: 255 if p > 175 else 0)
            variants = [prepared, binarized, img]
            configs = [
                '--oem 3 --psm 6',
                '--oem 3 --psm 11',
                '--oem 1 --psm 4',
            ]
            best = ''
            for variant in variants:
                for cfg in configs:
                    try:
                        out = pytesseract.image_to_string(variant, timeout=timeout, config=cfg)
                    except TypeError:
                        out = pytesseract.image_to_string(variant, config=cfg)
                    except Exception:
                        continue
                    if out and len(out.strip()) > len(best.strip()):
                        best = out
            return best
        except Exception:
            return ''

    try:
        if ext == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()[:EXTRACT_MAX_TEXT_BYTES]  # limit text to configured max

        if ext == 'pdf':
            try:
                texts = []
                with pdfplumber.open(file_path) as pdf:
                    # Only process first N pages for text extraction
                    for i, page in enumerate(pdf.pages[:EXTRACT_PDF_MAX_PAGES]):
                        if i >= EXTRACT_PDF_MAX_PAGES:
                            break
                        page_text = page.extract_text()
                        if page_text:
                            texts.append(page_text)
                combined = "\n".join(texts).strip()
                if combined:
                    return combined[:EXTRACT_MAX_TEXT_BYTES]  # limit to configured max
            except Exception as e:
                logger.debug("pdfplumber error: %s", e)

            # Only do OCR if explicitly enabled (very slow!)
            if not enable_ocr:
                logger.debug("OCR disabled for %s (pdfplumber found no text)", file_path)
                return ""  # Return empty instead of doing slow OCR

            try:
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"OCR timeout after {timeout_seconds}s")
                
                # Convert only first 3 pages to images (not entire document)
                images = convert_from_path(file_path, dpi=ocr_dpi, first_page=1, last_page=min(3, 999))
            except Exception as e:
                logger.debug("pdf2image error: %s", e)
                return ""
            ocr_texts = []
            for i, img in enumerate(images):
                if i >= max_pages_for_ocr:
                    break
                try:
                    txt = _ocr_with_fallback(img, timeout=timeout_seconds)
                    if txt and txt.strip():
                        ocr_texts.append(txt)
                except Exception as e:
                    logger.debug("pytesseract error on page %d: %s", i, e)
            return ("\n".join(ocr_texts).strip())[:EXTRACT_MAX_TEXT_BYTES]  # limit to configured max

        if ext in ('docx', 'doc'):
            try:
                document = docx.Document(file_path)
                paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
                return ("\n".join(paragraphs).strip())[:EXTRACT_MAX_TEXT_BYTES]  # limit to configured max
            except Exception as e:
                logger.debug("docx error: %s", e)
                return ""

        if ext in ('png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'webp'):
            if not enable_ocr:
                logger.debug("OCR disabled for image file %s", file_path)
                return ""
            try:
                img = Image.open(file_path)
                text = _ocr_with_fallback(img, timeout=timeout_seconds)
                return (text or "")[:EXTRACT_MAX_TEXT_BYTES]
            except Exception as e:
                logger.debug("image OCR error: %s", e)
                return ""

    except Exception as e:
        logger.debug("extract_text general error: %s", e)
        return ""

    return ""


def classify_document_with_confidence(text):
    t = (text or "").lower()
    if _ml_pipeline is not None:
        try:
            pred = _ml_pipeline.predict([text or ""])[0]
            if hasattr(_ml_pipeline, "predict_proba"):
                probs = _ml_pipeline.predict_proba([text or ""])[0]
                classes = list(getattr(_ml_pipeline, 'classes_', []))
                indexed = sorted(
                    [(classes[i] if i < len(classes) else f'class_{i}', float(p)) for i, p in enumerate(probs)],
                    key=lambda item: item[1],
                    reverse=True,
                )
                best_class = str(pred)
                best_conf = float(max(probs))
                top_candidates = ', '.join(f"{name}:{score:.2f}" for name, score in indexed[:2])
                if best_conf >= 0.45:
                    return best_class, round(best_conf, 4), top_candidates
            else:
                return str(pred), 0.7, f"{pred}:0.70"
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

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best = ranked[0][0]
    best_score = ranked[0][1]
    top_candidates = ', '.join(f"{name}:{score}" for name, score in ranked[:2])
    if best_score >= 2:
        confidence = min(0.65, 0.35 + (best_score * 0.05))
        return best, round(confidence, 4), top_candidates
    return 'other', 0.2, top_candidates


def classify_document(text):
    category, _, _ = classify_document_with_confidence(text)
    return category


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


def score_resume_with_ranker(text, prefs, runtime):
    base_score, base_note = score_resume(text, prefs)
    if not runtime.get('enable_resume_ranker'):
        return base_score, base_note, None, 'disabled'

    if _ml_resume_ranker is None:
        return base_score, f"{base_note}; ranker_unavailable", None, RESUME_RANKER_MODEL_VERSION

    try:
        ranker_score = None
        ranker_conf = None
        if hasattr(_ml_resume_ranker, 'predict_proba'):
            probs = _ml_resume_ranker.predict_proba([text or ""])[0]
            ranker_conf = float(max(probs)) if len(probs) > 0 else 0.0
            if len(probs) >= 2:
                ranker_score = float(probs[1]) * 100.0
            elif len(probs) == 1:
                ranker_score = float(probs[0]) * 100.0
        else:
            pred = _ml_resume_ranker.predict([text or ""])[0]
            ranker_score = float(pred)
            if 0.0 <= ranker_score <= 1.0:
                ranker_score *= 100.0
            ranker_conf = 0.6

        if ranker_score is None:
            return base_score, f"{base_note}; ranker_no_score", ranker_conf, RESUME_RANKER_MODEL_VERSION

        ranker_score = max(0.0, min(100.0, ranker_score))
        blended = round((0.7 * base_score) + (0.3 * ranker_score), 2)
        note = f"{base_note}; ranker:{ranker_score:.1f}"
        return blended, note, ranker_conf, RESUME_RANKER_MODEL_VERSION
    except Exception as exc:
        logger.warning('Resume ranker fallback to rule-based score: %s', exc)
        return base_score, f"{base_note}; ranker_error", None, RESUME_RANKER_MODEL_VERSION


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


def log_upload(filename, saved_path, summary, category, uploader_email=None, resume_score=None, resume_rank_note=None, resume_risk_score=None, resume_risk_flags=None, resume_fit_score=None, resume_fit_confidence=None, ml_confidence=None, top_candidates=None, model_version=None, classifier_version=None, ranker_version=None, processing_status='completed', processing_error=None):
    conn = sqlite3.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute(
        'INSERT INTO uploads (filename, saved_path, summary, category, uploader_email, uploaded_at, resume_score, resume_rank_note, resume_risk_score, resume_risk_flags, resume_fit_score, resume_fit_confidence, ml_confidence, top_candidates, model_version, classifier_version, ranker_version, processing_status, processing_error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
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
            resume_fit_score,
            resume_fit_confidence,
            ml_confidence,
            top_candidates,
            model_version,
            classifier_version,
            ranker_version,
            processing_status,
            processing_error,
        )
    )
    conn.commit()
    conn.close()


def create_upload_pending(filename, saved_path, uploader_email=None):
    conn = sqlite3.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute(
          '''INSERT INTO uploads (filename, saved_path, summary, category, uploader_email, uploaded_at, model_version, classifier_version, ranker_version, processing_status)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id''',
        (
            filename,
            saved_path,
            'Processing started',
            'pending',
            uploader_email,
            datetime.utcnow().isoformat(),
            MODEL_VERSION,
            CLASSIFIER_MODEL_VERSION,
            RESUME_RANKER_MODEL_VERSION,
            'queued',
        ),
    )
    row = c.fetchone()
    conn.commit()
    conn.close()
    return row[0] if row else None


def update_upload_record(upload_id, **updates):
    if not upload_id or not updates:
        return
    allowed = {
        'summary', 'category', 'resume_score', 'resume_rank_note', 'resume_risk_score', 'resume_risk_flags',
        'resume_fit_score', 'resume_fit_confidence',
        'ml_confidence', 'top_candidates', 'model_version', 'classifier_version', 'ranker_version',
        'processing_status', 'processing_error'
    }
    fields = []
    params = []
    for key, value in updates.items():
        if key in allowed:
            fields.append(f"{key} = ?")
            params.append(value)
    if not fields:
        return
    params.append(upload_id)
    conn = sqlite3.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute(f"UPDATE uploads SET {', '.join(fields)} WHERE id = ?", params)
    conn.commit()
    conn.close()


def record_review_feedback(upload_id, admin_action, from_category, to_category, shortlist_fit=None, extraction_feedback=None, reviewer_note=None):
    user = get_current_user()
    actor = user.get('username') if user else 'system'
    conn = sqlite3.connect(DATABASE_URL)
    try:
        c = conn.cursor()
        c.execute(
            '''INSERT INTO review_feedback (
                   upload_id, admin_action, from_category, to_category, shortlist_fit,
                   extraction_feedback, reviewer_note, actor, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                upload_id,
                admin_action,
                from_category,
                to_category,
                shortlist_fit,
                extraction_feedback,
                reviewer_note,
                actor,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    except Exception as e:
        logger.warning('Review feedback save failed: %s', e)
    finally:
        conn.close()


def get_latest_upload_by_filename(filename):
    conn = sqlite3.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute(
        '''SELECT id, filename, category, summary, processing_status, ml_confidence, top_candidates,
                  processing_error, resume_score, resume_risk_score, resume_risk_flags
           FROM uploads WHERE filename = ?
           ORDER BY uploaded_at DESC LIMIT 1''',
        (filename,),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        'id': row[0],
        'filename': row[1],
        'category': row[2],
        'summary': row[3],
        'processing_status': row[4],
        'ml_confidence': row[5],
        'top_candidates': row[6],
        'processing_error': row[7],
        'resume_score': row[8],
        'resume_risk_score': row[9],
        'resume_risk_flags': row[10],
    }


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


def process_document_in_background(file_path, filename, uploader_email, upload_id=None):
    """
    Process a document in the background (extract, classify, summarize, log).
    This runs in a separate thread to avoid blocking the upload response.
    """
    try:
        logger.info("🔄 Background processing started for: %s", filename)
        runtime = get_runtime_settings()
        
        update_upload_record(upload_id, processing_status='extracting')
        # Extract text (OCR setting from ENABLE_OCR env var)
        text = extract_text(file_path, enable_ocr=ENABLE_OCR)
        
        # Summarize
        summary = simple_summarize(text)
        
        update_upload_record(upload_id, processing_status='classifying')
        category, confidence, top_candidates = classify_document_with_confidence(text)
        threshold = get_auto_route_threshold(category, runtime)
        requires_review = confidence < threshold
        if requires_review:
            category = 'review_required'
        
        # Resume scoring if applicable
        resume_score = None
        resume_rank_note = None
        resume_risk_score = None
        resume_risk_flags = None
        resume_fit_score = None
        resume_fit_confidence = None
        ranker_version = RESUME_RANKER_MODEL_VERSION
        
        if category == 'resume':
            prefs = parse_resume_preferences(runtime)
            resume_score, resume_rank_note, resume_fit_confidence, ranker_version = score_resume_with_ranker(text, prefs, runtime)
            resume_fit_score = resume_score
            resume_risk_score, resume_risk_flags = assess_resume_risk(text)

        status_value = 'review_required' if requires_review else 'completed'
        processing_error = None
        if requires_review:
            processing_error = f'Low classification confidence ({confidence:.2f}) below threshold ({threshold:.2f}); held for manual review.'

        update_upload_record(
            upload_id,
            summary=summary,
            category=category,
            resume_score=resume_score,
            resume_rank_note=resume_rank_note,
            resume_risk_score=resume_risk_score,
            resume_risk_flags=', '.join(resume_risk_flags or []) if isinstance(resume_risk_flags, list) else resume_risk_flags,
            resume_fit_score=resume_fit_score,
            resume_fit_confidence=resume_fit_confidence,
            ml_confidence=confidence,
            top_candidates=top_candidates,
            model_version=MODEL_VERSION,
            classifier_version=CLASSIFIER_MODEL_VERSION,
            ranker_version=ranker_version,
            processing_status=status_value,
            processing_error=processing_error,
        )
        
        # Optional local routing output copy
        if runtime.get('route_local_enabled') and not requires_review:
            route_dir = get_route_output_dir(category, runtime)
            if route_dir:
                try:
                    os.makedirs(route_dir, exist_ok=True)
                    routed_path = os.path.join(route_dir, filename)
                    shutil.copy2(file_path, routed_path)
                except Exception as e:
                    logger.warning('Failed to copy routed file to %s: %s', route_dir, e)
        
        # Optional auto-reply to uploader
        if uploader_email:
            from_name = runtime.get('from_name') or FROM_NAME
            reply_subject = f"Receipt: {filename} (classified: {category})"
            reply_body = f"Hi,\n\nWe processed your file '{filename}'.\nCategory: {category}\nSummary:\n{summary}\n\nThanks,\n{from_name}"
            try:
                send_email_with_attachment(uploader_email, reply_subject, reply_body, None, None)
            except Exception as e:
                logger.warning("Auto-reply failed for %s: %s", uploader_email, e)
        
        logger.info("✅ Background processing completed for: %s (category: %s)", filename, category)
        
    except Exception as e:
        update_upload_record(upload_id, processing_status='failed', processing_error=str(e))
        logger.error("❌ Background processing failed for %s: %s", filename, e)

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
    """
    Optimized upload endpoint:
    - Saves files immediately (fast)
    - Moves to category folders immediately
    - Returns success page immediately
    - Processing (extraction, classification, etc.) happens in background threads
    """
    if 'file' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('file')
    uploader_email = request.form.get('email', None)

    if not files or all(f.filename == '' for f in files):
        return redirect(request.url)

    runtime = get_runtime_settings()
    upload_root = _normalize_dir(runtime.get('upload_folder')) or os.path.abspath(UPLOAD_FOLDER)
    app.config['UPLOAD_FOLDER'] = upload_root
    
    results = []  # For display in response (minimal info)
    batch_sent_info = []

    # ===== PHASE 1: Save files immediately (fast) =====
    os.makedirs(upload_root, exist_ok=True)
    saved_files = []
    
    for file in files:
        if not (file and allowed_file(file.filename)):
            continue

        filename = secure_filename(file.filename)
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        saved_path = os.path.join(upload_root, saved_filename)
        
        # Save file
        file.save(saved_path)
        
        # Quick classification guess (keywords only, no ML) for folder routing
        with open(saved_path, 'rb') as f:
            try:
                # Quick text peek for classification
                peek_text = f.read(5000).decode('utf-8', errors='ignore').lower()
            except:
                peek_text = ""
        
        # Simple keyword-based quick classification for folder
        if 'invoice' in peek_text or 'amount due' in peek_text:
            quick_category = 'invoice'
        elif 'payslip' in peek_text or 'salary' in peek_text:
            quick_category = 'payslip'
        elif 'purchase order' in peek_text or 'po no' in peek_text:
            quick_category = 'purchase_order'
        elif 'minutes' in peek_text or 'agenda' in peek_text:
            quick_category = 'minutes'
        elif 'resume' in peek_text or 'curriculum vitae' in peek_text:
            quick_category = 'resume'
        else:
            quick_category = 'other'
        
        # Create category folder and move file (fast operation)
        cat_folder = "".join(ch for ch in quick_category if ch.isalnum() or ch in ('_', '-')).lower() or 'other'
        target_dir = os.path.join(upload_root, cat_folder)
        os.makedirs(target_dir, exist_ok=True)
        dest_path = os.path.join(target_dir, saved_filename)
        
        try:
            shutil.move(saved_path, dest_path)
            saved_path = dest_path
        except Exception as e:
            logger.warning("Failed to move file to category folder: %s", e)
        
        saved_files.append({
            'filename': saved_filename,
            'saved_path': saved_path,
            'uploader_email': uploader_email,
            'quick_category': quick_category,
            'upload_id': create_upload_pending(saved_filename, saved_path, uploader_email),
        })
        
        results.append({
            'filename': saved_filename,
            'category': quick_category,
            'summary': 'Queued for processing...',
            'processing_status': 'queued',
            'ml_confidence': None,
            'top_candidates': '-',
            'saved_path': saved_path,
            'route': route_for_category(quick_category),
        })

    if not saved_files:
        return "No valid files uploaded.", 400

    # ===== PHASE 2: Start background processing threads =====
    # Process each file in background threads (extraction, classification, etc.)
    for file_info in saved_files:
        thread = threading.Thread(
            target=process_document_in_background,
            args=(file_info['saved_path'], file_info['filename'], file_info['uploader_email'], file_info.get('upload_id')),
            daemon=True
        )
        thread.start()
    
    # ===== PHASE 3: Start background email forwarding thread =====
    # Email batch forwarding happens asynchronously
    if runtime.get('route_email_enabled'):
        thread = threading.Thread(
            target=_send_batch_emails_async,
            args=(saved_files, runtime),
            daemon=True
        )
        thread.start()

    # ===== Return immediately without waiting for processing =====
    return render_template('result_batch.html', results=results)


@app.route('/ingest/webhook/email', methods=['POST'])
def webhook_ingest_email():
    runtime = get_runtime_settings()
    inbound_adapter = (runtime.get('inbound_adapter') or INBOUND_ADAPTER).lower()
    if inbound_adapter not in {'webhook', 'hybrid'}:
        return jsonify({'ok': False, 'error': 'Webhook intake is disabled by inbound adapter mode.'}), 403

    expected_secret = (runtime.get('webhook_shared_secret') or WEBHOOK_SHARED_SECRET or '').strip()
    provided_secret = (request.headers.get('X-Webhook-Secret') or request.args.get('secret') or '').strip()
    if expected_secret and provided_secret != expected_secret:
        return jsonify({'ok': False, 'error': 'Unauthorized webhook secret.'}), 401

    files = request.files.getlist('file')
    if not files:
        # Fallback for generic multipart payloads with arbitrary field names.
        files = list(request.files.values())

    uploader_email = (request.form.get('sender') or request.form.get('uploader_email') or request.form.get('email') or '').strip() or None
    source_label = 'webhook-multipart'

    json_attachments = []
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        normalized = _normalize_webhook_json_payload(payload)
        json_attachments = normalized.get('attachments') or []
        uploader_email = uploader_email or normalized.get('sender')
        source_label = normalized.get('provider') or 'generic-json'

    if (not files or all((not f) or (not f.filename) for f in files)) and not json_attachments:
        return jsonify({'ok': False, 'error': 'No files found in webhook payload. Provide multipart files or JSON base64 attachments.'}), 400

    upload_root = _normalize_dir(runtime.get('upload_folder')) or os.path.abspath(UPLOAD_FOLDER)
    app.config['UPLOAD_FOLDER'] = upload_root
    os.makedirs(upload_root, exist_ok=True)

    results = []
    saved_files = []
    for file in files:
        if not (file and file.filename and allowed_file(file.filename)):
            continue

        filename = secure_filename(file.filename)
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        saved_path = os.path.join(upload_root, saved_filename)
        file.save(saved_path)

        queued = _save_and_queue_file(saved_filename, saved_path, upload_root, uploader_email)
        saved_files.append({
            'filename': queued['filename'],
            'saved_path': queued['saved_path'],
            'uploader_email': queued['uploader_email'],
            'upload_id': queued['upload_id'],
        })
        results.append({
            'upload_id': queued['upload_id'],
            'filename': queued['filename'],
            'quick_category': queued['quick_category'],
            'status': 'queued',
        })

    # Provider-style JSON attachments (Gmail/Graph-like) with base64 content.
    for item in json_attachments:
        filename = secure_filename(item.get('filename') or '')
        if not filename:
            continue
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        saved_path = os.path.join(upload_root, saved_filename)
        try:
            with open(saved_path, 'wb') as f:
                f.write(item.get('content') or b'')
        except Exception as e:
            logger.warning('Webhook JSON attachment save failed for %s: %s', saved_filename, e)
            continue

        queued = _save_and_queue_file(saved_filename, saved_path, upload_root, uploader_email)
        saved_files.append({
            'filename': queued['filename'],
            'saved_path': queued['saved_path'],
            'uploader_email': queued['uploader_email'],
            'upload_id': queued['upload_id'],
        })
        results.append({
            'upload_id': queued['upload_id'],
            'filename': queued['filename'],
            'quick_category': queued['quick_category'],
            'status': 'queued',
        })

    if not saved_files:
        return jsonify({'ok': False, 'error': 'No supported document files found.'}), 400

    for file_info in saved_files:
        thread = threading.Thread(
            target=process_document_in_background,
            args=(file_info['saved_path'], file_info['filename'], file_info['uploader_email'], file_info.get('upload_id')),
            daemon=True,
        )
        thread.start()

    if runtime.get('route_email_enabled'):
        thread = threading.Thread(
            target=_send_batch_emails_async,
            args=(saved_files, runtime),
            daemon=True,
        )
        thread.start()

    return jsonify({'ok': True, 'source': source_label, 'queued': len(saved_files), 'items': results}), 202


def _send_batch_emails_async(saved_files, runtime):
    """Send batch emails in background (after file processing completes)."""
    try:
        # Give processing threads time to complete (max 30 seconds)
        logger.info("⏳ Waiting for document processing before sending batch emails...")
        time.sleep(5)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        upload_root = _normalize_dir(runtime.get('upload_folder')) or os.path.abspath(UPLOAD_FOLDER)
        
        # Query processed documents and group by recipient
        from collections import defaultdict
        batch_map = defaultdict(list)
        
        for file_info in saved_files:
            try:
                conn = get_db_conn()
                c = conn.cursor()
                c.execute(
                    'SELECT filename, category, summary, saved_path, processing_status FROM uploads WHERE filename = ? ORDER BY uploaded_at DESC LIMIT 1',
                    (file_info['filename'],)
                )
                row = c.fetchone()
                conn.close()
                
                if row and row[4] == 'completed':
                    category = row[1]
                    forward_to = route_for_category(category)
                    if forward_to:
                        batch_map[forward_to].append({
                            'filename': row[0],
                            'category': row[1],
                            'summary': row[2] or '',
                            'saved_path': row[3],
                        })
            except Exception as e:
                logger.warning("Failed to fetch processed file info: %s", e)
        
        # Send batch emails
        for recipient, items in batch_map.items():
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
                logger.warning("Failed to write batch CSV: %s", e)
                continue

            attachment_paths = [csv_path]
            attachment_names = [os.path.basename(csv_path)]

            # Attach actual files
            for it in items:
                p = it.get('saved_path')
                if p and os.path.exists(p) and os.path.isfile(p):
                    attachment_paths.append(p)
                    attachment_names.append(os.path.basename(p))

            subj = f"[Batch Forwarded Files] {len(items)} files - {timestamp}"
            body_lines = [
                "Hello,",
                "",
                "This is an automated batch of files classified and routed to you.",
                "",
                "Files included:"
            ]
            for it in items:
                body_lines.append(f"- {it.get('filename')} (category: {it.get('category')})")
            body_lines.extend(["", "Both the original files and a CSV summary are attached."])
            body = "\n".join(body_lines)

            logger.info("📤 Sending batch email to %s with %d files...", recipient, len(items))
            send_email_with_attachments(recipient, subj, body, attachment_paths=attachment_paths, attachment_names=attachment_names)
            
            # Cleanup temp CSV
            try:
                if os.path.exists(csv_path):
                    os.remove(csv_path)
            except Exception as e:
                logger.warning("Failed to delete temp CSV: %s", e)
                
    except Exception as e:
        logger.error("❌ Batch email sending failed: %s", e)

@app.route('/chat')
@login_required
def chat_page():
    return render_template('chat.html')


@app.route('/upload/status', methods=['GET'])
@login_required
def upload_status_batch():
    raw = (request.args.get('filenames') or '').strip()
    if not raw:
        return jsonify({'items': []})

    filenames = [f.strip() for f in raw.split(',') if f.strip()]
    payload = []
    for filename in filenames[:200]:
        row = get_latest_upload_by_filename(filename)
        if row:
            payload.append(row)
    return jsonify({'items': payload})


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
        SELECT id, filename, saved_path, summary, category, uploader_email, uploaded_at, resume_score, resume_risk_score, resume_risk_flags,
               ml_confidence, top_candidates, model_version, processing_status, processing_error
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
    c.execute(f"SELECT id, filename, saved_path, summary, category, uploader_email, uploaded_at, resume_score, resume_risk_score, resume_risk_flags, ml_confidence, top_candidates, model_version, processing_status, processing_error FROM uploads {where_sql} ORDER BY uploaded_at DESC", params)
    rows = c.fetchall()
    conn.close()

    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(['id','filename','saved_path','category','uploader_email','uploaded_at','summary','resume_score','resume_risk_score','resume_risk_flags','ml_confidence','top_candidates','model_version','processing_status','processing_error'])
    for r in rows:
        writer.writerow([r[0], r[1], r[2] or '', r[4] or '', r[5] or '', r[6] or '', r[3] or '', r[7] or '', r[8] or '', r[9] or '', r[10] or '', r[11] or '', r[12] or '', r[13] or '', r[14] or ''])
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
        c.execute("SELECT COUNT(*) FROM uploads WHERE processing_status IN ('review_required', 'failed')")
        review_queue_count = c.fetchone()[0]
    finally:
        conn.close()

    runtime = get_runtime_settings()
    return render_template(
        'admin.html',
        total_users=total_users,
        total_admins=total_admins,
        active_users=active_users,
        review_queue_count=review_queue_count,
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
        'IMAP_SKIP_EXISTING_UNSEEN': '1' if _bool_from_str(runtime.get('imap_skip_existing_unseen'), default=True) else '0',
        'IMAP_ALLOWED_SENDERS': runtime.get('imap_allowed_senders') or '',
        'IMAP_SUBJECT_KEYWORD': runtime.get('imap_subject_keyword') or '',
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
        'ENABLE_RESUME_RANKER': '1' if runtime.get('enable_resume_ranker') else '0',
        'RANKER_MIN_FIT_SCORE': str(runtime.get('ranker_min_fit_score') if runtime.get('ranker_min_fit_score') is not None else RANKER_MIN_FIT_SCORE),
        'CLASSIFIER_MODEL_PATH': runtime.get('classifier_model_path') or CLASSIFIER_MODEL_PATH,
        'RESUME_RANKER_MODEL_PATH': runtime.get('resume_ranker_model_path') or RESUME_RANKER_MODEL_PATH,
        'AUTO_ROUTE_CONFIDENCE_THRESHOLD': str(runtime.get('auto_route_confidence_threshold') or AUTO_ROUTE_CONFIDENCE_THRESHOLD),
        'AUTO_ROUTE_THRESHOLD_INVOICE': str(runtime.get('auto_route_threshold_invoice') or ''),
        'AUTO_ROUTE_THRESHOLD_PAYSLIP': str(runtime.get('auto_route_threshold_payslip') or ''),
        'AUTO_ROUTE_THRESHOLD_PURCHASE_ORDER': str(runtime.get('auto_route_threshold_purchase_order') or ''),
        'AUTO_ROUTE_THRESHOLD_MINUTES': str(runtime.get('auto_route_threshold_minutes') or ''),
        'AUTO_ROUTE_THRESHOLD_RESUME': str(runtime.get('auto_route_threshold_resume') or ''),
        'AUTO_ROUTE_THRESHOLD_OTHER': str(runtime.get('auto_route_threshold_other') or ''),
        'INBOUND_ADAPTER': (runtime.get('inbound_adapter') or INBOUND_ADAPTER).lower(),
    }
    has_email_pass = bool(runtime.get('email_pass'))
    has_imap_pass = bool(runtime.get('imap_pass'))
    has_webhook_secret = bool(runtime.get('webhook_shared_secret'))

    if request.method == 'POST':
        form = request.form

        smtp_port = _safe_int(form.get('SMTP_PORT', '').strip(), -1)
        imap_port = _safe_int(form.get('IMAP_PORT', '').strip(), -1)

        if not (1 <= smtp_port <= 65535):
            flash('SMTP port must be between 1 and 65535.', 'error')
            return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass, has_webhook_secret=has_webhook_secret)
        if not (1 <= imap_port <= 65535):
            flash('IMAP port must be between 1 and 65535.', 'error')
            return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass, has_webhook_secret=has_webhook_secret)

        auto_threshold = _safe_float(form.get('AUTO_ROUTE_CONFIDENCE_THRESHOLD', '').strip(), -1)
        if not (0.0 <= auto_threshold <= 1.0):
            flash('Auto-route confidence threshold must be between 0.0 and 1.0.', 'error')
            return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass, has_webhook_secret=has_webhook_secret)

        ranker_min_fit_score = _safe_float(form.get('RANKER_MIN_FIT_SCORE', '').strip(), -1)
        if not (0.0 <= ranker_min_fit_score <= 100.0):
            flash('Ranker minimum fit score must be between 0 and 100.', 'error')
            return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass, has_webhook_secret=has_webhook_secret)

        inbound_adapter = (form.get('INBOUND_ADAPTER') or '').strip().lower()
        if inbound_adapter not in {'imap', 'webhook', 'hybrid'}:
            flash('Inbound adapter must be one of: imap, webhook, hybrid.', 'error')
            return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass, has_webhook_secret=has_webhook_secret)

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
                return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass, has_webhook_secret=has_webhook_secret)

        updates = {
            'email_user': email_user,
            'smtp_server': (form.get('SMTP_SERVER') or '').strip(),
            'smtp_port': str(smtp_port),
            'imap_host': (form.get('IMAP_HOST') or '').strip(),
            'imap_port': str(imap_port),
            'imap_user': imap_user,
            'imap_skip_existing_unseen': '1' if form.get('IMAP_SKIP_EXISTING_UNSEEN') == 'on' else '0',
            'imap_allowed_senders': (form.get('IMAP_ALLOWED_SENDERS') or '').strip(),
            'imap_subject_keyword': (form.get('IMAP_SUBJECT_KEYWORD') or '').strip(),
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
            'enable_resume_ranker': '1' if form.get('ENABLE_RESUME_RANKER') == 'on' else '0',
            'ranker_min_fit_score': str(ranker_min_fit_score),
            'classifier_model_path': (form.get('CLASSIFIER_MODEL_PATH') or '').strip() or CLASSIFIER_MODEL_PATH,
            'resume_ranker_model_path': (form.get('RESUME_RANKER_MODEL_PATH') or '').strip() or RESUME_RANKER_MODEL_PATH,
            'auto_route_confidence_threshold': str(auto_threshold),
            'auto_route_threshold_invoice': (form.get('AUTO_ROUTE_THRESHOLD_INVOICE') or '').strip(),
            'auto_route_threshold_payslip': (form.get('AUTO_ROUTE_THRESHOLD_PAYSLIP') or '').strip(),
            'auto_route_threshold_purchase_order': (form.get('AUTO_ROUTE_THRESHOLD_PURCHASE_ORDER') or '').strip(),
            'auto_route_threshold_minutes': (form.get('AUTO_ROUTE_THRESHOLD_MINUTES') or '').strip(),
            'auto_route_threshold_resume': (form.get('AUTO_ROUTE_THRESHOLD_RESUME') or '').strip(),
            'auto_route_threshold_other': (form.get('AUTO_ROUTE_THRESHOLD_OTHER') or '').strip(),
            'inbound_adapter': inbound_adapter,
        }

        threshold_keys = [
            'auto_route_threshold_invoice',
            'auto_route_threshold_payslip',
            'auto_route_threshold_purchase_order',
            'auto_route_threshold_minutes',
            'auto_route_threshold_resume',
            'auto_route_threshold_other',
        ]
        for key in threshold_keys:
            raw = updates.get(key)
            if raw == '':
                continue
            value = _safe_float(raw, -1)
            if not (0.0 <= value <= 1.0):
                flash(f'{key} must be between 0.0 and 1.0 or left blank.', 'error')
                return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass, has_webhook_secret=has_webhook_secret)
            updates[key] = str(value)

        path_fields = [
            'upload_folder', 'route_dir_invoice', 'route_dir_payslip',
            'route_dir_purchase_order', 'route_dir_minutes', 'route_dir_resume', 'log_file_path'
        ]
        for key in path_fields:
            p = _normalize_dir(updates.get(key))
            if not p:
                flash(f'{key} is required.', 'error')
                return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass, has_webhook_secret=has_webhook_secret)
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
            return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass, has_webhook_secret=has_webhook_secret)

        email_pass = (form.get('EMAIL_PASS') or '').strip()
        if email_pass:
            updates['email_pass'] = email_pass
        imap_pass = (form.get('IMAP_PASS') or '').strip()
        if imap_pass:
            updates['imap_pass'] = imap_pass
        webhook_secret = (form.get('WEBHOOK_SHARED_SECRET') or '').strip()
        if webhook_secret:
            updates['webhook_shared_secret'] = webhook_secret

        try:
            for key, value in updates.items():
                set_setting(DATABASE_URL, key, value, encrypt=key in SENSITIVE_SETTING_KEYS)
        except Exception as e:
            flash(f'Failed to save settings: {e}', 'error')
            return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass, has_webhook_secret=has_webhook_secret)

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

    return render_template('settings.html', values=display, has_email_pass=has_email_pass, has_imap_pass=has_imap_pass, has_webhook_secret=has_webhook_secret)


@app.route('/admin/review', methods=['GET', 'POST'])
@admin_required
def admin_review_queue():
    if request.method == 'POST':
        action = (request.form.get('action') or '').strip().lower()
        selected_ids = [_safe_int(x, -1) for x in request.form.getlist('upload_ids')]
        selected_ids = [x for x in selected_ids if x > 0]
        if action in {'bulk_retry', 'bulk_approve', 'bulk_relabel'}:
            if not selected_ids:
                flash('Select at least one review item for bulk action.', 'error')
                return redirect(url_for('admin_review_queue'))

            bulk_action = 'retry' if action == 'bulk_retry' else ('approve' if action == 'bulk_approve' else 'relabel')
            bulk_category = (request.form.get('bulk_category') or '').strip().lower()
            bulk_shortlist_fit = (request.form.get('bulk_shortlist_fit') or '').strip().lower() or None
            bulk_extraction_feedback = (request.form.get('bulk_extraction_feedback') or '').strip() or None
            bulk_reviewer_note = (request.form.get('bulk_reviewer_note') or '').strip() or None
            if bulk_action in {'approve', 'relabel'} and bulk_category not in CLASSIFICATION_CATEGORIES:
                flash('Choose a valid category for bulk approve/relabel.', 'error')
                return redirect(url_for('admin_review_queue'))

            processed = 0
            for rid in selected_ids:
                conn = get_db_conn()
                try:
                    c = conn.cursor()
                    c.execute(
                        '''SELECT id, filename, saved_path, summary, category, uploader_email, processing_status
                           FROM uploads WHERE id = ?''',
                        (rid,),
                    )
                    row = c.fetchone()
                finally:
                    conn.close()
                if not row:
                    continue

                _, filename, saved_path, summary, current_category, uploader_email, current_status = row
                runtime = get_runtime_settings()

                if bulk_action == 'retry':
                    update_upload_record(rid, processing_status='queued', processing_error=None)
                    thread = threading.Thread(
                        target=process_document_in_background,
                        args=(saved_path, filename, uploader_email, rid),
                        daemon=True,
                    )
                    thread.start()
                    audit_log('review_retry', f'upload_id={rid}, filename={filename}, bulk=1')
                    processed += 1
                    continue

                resume_score = None
                resume_rank_note = None
                resume_risk_score = None
                resume_risk_flags = None
                if bulk_category == 'resume' and saved_path and os.path.exists(saved_path):
                    text = extract_text(saved_path, enable_ocr=ENABLE_OCR)
                    prefs = parse_resume_preferences(runtime)
                    resume_score, resume_rank_note = score_resume(text, prefs)
                    resume_risk_score, resume_risk_flags = assess_resume_risk(text)

                update_upload_record(
                    rid,
                    category=bulk_category,
                    resume_score=resume_score,
                    resume_rank_note=resume_rank_note,
                    resume_risk_score=resume_risk_score,
                    resume_risk_flags=', '.join(resume_risk_flags or []) if isinstance(resume_risk_flags, list) else resume_risk_flags,
                    processing_status='completed',
                    processing_error=None,
                )

                if runtime.get('route_local_enabled') and saved_path and os.path.exists(saved_path):
                    route_dir = get_route_output_dir(bulk_category, runtime)
                    if route_dir:
                        try:
                            os.makedirs(route_dir, exist_ok=True)
                            shutil.copy2(saved_path, os.path.join(route_dir, os.path.basename(saved_path)))
                        except Exception as e:
                            logger.warning('Bulk manual routing copy failed for upload_id=%s: %s', rid, e)

                if runtime.get('route_email_enabled') and saved_path and os.path.exists(saved_path):
                    recipient = route_for_category(bulk_category)
                    if recipient:
                        subj = f"[Bulk Manual Review Approved] {filename} ({bulk_category})"
                        body = (
                            f"The document '{filename}' was approved from review queue.\n"
                            f"Category: {bulk_category}\n"
                            f"Summary: {summary or 'N/A'}"
                        )
                        try:
                            send_email_with_attachment(recipient, subj, body, saved_path, os.path.basename(saved_path))
                        except Exception as e:
                            logger.warning('Bulk manual routing email failed for upload_id=%s: %s', rid, e)

                audit_log(
                    'review_approved' if bulk_action == 'approve' else 'review_relabel',
                    f'upload_id={rid}, from={current_category}, to={bulk_category}, previous_status={current_status}, bulk=1',
                )
                record_review_feedback(
                    rid,
                    'bulk_approve' if bulk_action == 'approve' else 'bulk_relabel',
                    current_category,
                    bulk_category,
                    shortlist_fit=bulk_shortlist_fit,
                    extraction_feedback=bulk_extraction_feedback,
                    reviewer_note=bulk_reviewer_note,
                )
                processed += 1

            flash(f'Bulk action completed for {processed} item(s).', 'success')
            return redirect(url_for('admin_review_queue'))

        upload_id = _safe_int(request.form.get('upload_id', '').strip(), -1)
        selected_category = (request.form.get('category') or '').strip().lower()
        shortlist_fit = (request.form.get('shortlist_fit') or '').strip().lower() or None
        extraction_feedback = (request.form.get('extraction_feedback') or '').strip() or None
        reviewer_note = (request.form.get('reviewer_note') or '').strip() or None

        valid_shortlist_fit = {None, 'shortlist', 'reject', 'unsure'}
        if shortlist_fit not in valid_shortlist_fit:
            flash('Invalid shortlist feedback value.', 'error')
            return redirect(url_for('admin_review_queue'))

        if upload_id <= 0:
            flash('Invalid upload id.', 'error')
            return redirect(url_for('admin_review_queue'))

        conn = get_db_conn()
        try:
            c = conn.cursor()
            c.execute(
                '''SELECT id, filename, saved_path, summary, category, uploader_email, processing_status
                   FROM uploads WHERE id = ?''',
                (upload_id,),
            )
            row = c.fetchone()
        finally:
            conn.close()

        if not row:
            flash('Review item not found.', 'error')
            return redirect(url_for('admin_review_queue'))

        _, filename, saved_path, summary, current_category, uploader_email, current_status = row
        runtime = get_runtime_settings()

        if action == 'retry':
            update_upload_record(upload_id, processing_status='queued', processing_error=None)
            thread = threading.Thread(
                target=process_document_in_background,
                args=(saved_path, filename, uploader_email, upload_id),
                daemon=True,
            )
            thread.start()
            audit_log('review_retry', f'upload_id={upload_id}, filename={filename}')
            flash('Reprocessing started.', 'success')
            return redirect(url_for('admin_review_queue'))

        if action not in {'approve', 'relabel'}:
            flash('Unsupported review action.', 'error')
            return redirect(url_for('admin_review_queue'))

        if selected_category not in CLASSIFICATION_CATEGORIES:
            flash('Please choose a valid category.', 'error')
            return redirect(url_for('admin_review_queue'))

        resume_score = None
        resume_rank_note = None
        resume_risk_score = None
        resume_risk_flags = None
        if selected_category == 'resume' and saved_path and os.path.exists(saved_path):
            text = extract_text(saved_path, enable_ocr=ENABLE_OCR)
            prefs = parse_resume_preferences(runtime)
            resume_score, resume_rank_note = score_resume(text, prefs)
            resume_risk_score, resume_risk_flags = assess_resume_risk(text)

        # Apply admin decision and clear review error state.
        update_upload_record(
            upload_id,
            category=selected_category,
            resume_score=resume_score,
            resume_rank_note=resume_rank_note,
            resume_risk_score=resume_risk_score,
            resume_risk_flags=', '.join(resume_risk_flags or []) if isinstance(resume_risk_flags, list) else resume_risk_flags,
            processing_status='completed',
            processing_error=None,
        )

        # Route to local folder if enabled.
        if runtime.get('route_local_enabled') and saved_path and os.path.exists(saved_path):
            route_dir = get_route_output_dir(selected_category, runtime)
            if route_dir:
                try:
                    os.makedirs(route_dir, exist_ok=True)
                    shutil.copy2(saved_path, os.path.join(route_dir, os.path.basename(saved_path)))
                except Exception as e:
                    logger.warning('Manual routing copy failed for upload_id=%s: %s', upload_id, e)

        # Forward manually approved/re-labeled document via email if enabled.
        if runtime.get('route_email_enabled') and saved_path and os.path.exists(saved_path):
            recipient = route_for_category(selected_category)
            if recipient:
                subj = f"[Manual Review Approved] {filename} ({selected_category})"
                body = (
                    f"The document '{filename}' was approved from review queue.\n"
                    f"Category: {selected_category}\n"
                    f"Summary: {summary or 'N/A'}"
                )
                try:
                    send_email_with_attachment(recipient, subj, body, saved_path, os.path.basename(saved_path))
                except Exception as e:
                    logger.warning('Manual routing email failed for upload_id=%s: %s', upload_id, e)

        audit_log(
            'review_approved' if action == 'approve' else 'review_relabel',
            f'upload_id={upload_id}, from={current_category}, to={selected_category}, previous_status={current_status}',
        )
        record_review_feedback(
            upload_id,
            action,
            current_category,
            selected_category,
            shortlist_fit=shortlist_fit,
            extraction_feedback=extraction_feedback,
            reviewer_note=reviewer_note,
        )
        flash('Review decision saved and document routed.', 'success')
        return redirect(url_for('admin_review_queue'))

    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute(
            '''SELECT id, filename, category, summary, uploader_email, uploaded_at, ml_confidence,
                      top_candidates, processing_status, processing_error, saved_path
               FROM uploads
               WHERE processing_status IN ('review_required', 'failed')
               ORDER BY uploaded_at DESC
               LIMIT 300'''
        )
        rows = c.fetchall()
    finally:
        conn.close()

    return render_template('admin_review.html', rows=rows, categories=CLASSIFICATION_CATEGORIES)


@app.route('/admin/review/export', methods=['GET'])
@admin_required
def admin_review_export():
    action = (request.args.get('action') or '').strip().lower()
    from_category = (request.args.get('from_category') or '').strip().lower()
    to_category = (request.args.get('to_category') or '').strip().lower()
    shortlist_fit = (request.args.get('shortlist_fit') or '').strip().lower()

    valid_actions = {'approve', 'relabel', 'bulk_approve', 'bulk_relabel'}
    valid_shortlist_fit = {'shortlist', 'reject', 'unsure'}

    where_clauses = []
    params = []

    if action:
        if action not in valid_actions:
            flash('Invalid export action filter.', 'error')
            return redirect(url_for('admin_review_queue'))
        where_clauses.append('rf.admin_action = ?')
        params.append(action)

    if from_category:
        if from_category not in CLASSIFICATION_CATEGORIES and from_category not in {'review_required', 'failed', 'pending'}:
            flash('Invalid export from-category filter.', 'error')
            return redirect(url_for('admin_review_queue'))
        where_clauses.append('LOWER(COALESCE(rf.from_category, \"\")) = ?')
        params.append(from_category)

    if to_category:
        if to_category not in CLASSIFICATION_CATEGORIES:
            flash('Invalid export to-category filter.', 'error')
            return redirect(url_for('admin_review_queue'))
        where_clauses.append('LOWER(COALESCE(rf.to_category, \"\")) = ?')
        params.append(to_category)

    if shortlist_fit:
        if shortlist_fit not in valid_shortlist_fit:
            flash('Invalid export shortlist-fit filter.', 'error')
            return redirect(url_for('admin_review_queue'))
        where_clauses.append('LOWER(COALESCE(rf.shortlist_fit, \"\")) = ?')
        params.append(shortlist_fit)

    where_sql = ('WHERE ' + ' AND '.join(where_clauses)) if where_clauses else ''

    conn = get_db_conn()
    try:
        c = conn.cursor()
        c.execute(
            f'''
            SELECT rf.id, rf.upload_id, rf.admin_action, rf.from_category, rf.to_category,
                   rf.shortlist_fit, rf.extraction_feedback, rf.reviewer_note, rf.actor, rf.created_at,
                   u.filename, u.uploader_email, u.uploaded_at, u.category, u.ml_confidence, u.top_candidates
            FROM review_feedback rf
            LEFT JOIN uploads u ON u.id = rf.upload_id
            {where_sql}
            ORDER BY rf.created_at DESC
            ''',
            params,
        )
        rows = c.fetchall()
    finally:
        conn.close()

    si = StringIO()
    writer = csv.writer(si)
    writer.writerow([
        'feedback_id', 'upload_id', 'admin_action', 'from_category', 'to_category',
        'shortlist_fit', 'extraction_feedback', 'reviewer_note', 'actor', 'feedback_created_at',
        'filename', 'uploader_email', 'uploaded_at', 'current_upload_category', 'ml_confidence', 'top_candidates'
    ])
    for r in rows:
        writer.writerow([
            r[0], r[1], r[2] or '', r[3] or '', r[4] or '',
            r[5] or '', r[6] or '', r[7] or '', r[8] or '', r[9] or '',
            r[10] or '', r[11] or '', r[12] or '', r[13] or '', r[14] if r[14] is not None else '', r[15] or ''
        ])

    output = si.getvalue()
    si.close()

    audit_log('review_feedback_export', f'rows={len(rows)}, action={action or "all"}, from={from_category or "all"}, to={to_category or "all"}, shortlist_fit={shortlist_fit or "all"}')

    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    resp = Response(output, mimetype='text/csv')
    resp.headers['Content-Disposition'] = f'attachment; filename=review_feedback_{timestamp}.csv'
    return resp


@app.route('/admin/logs', methods=['GET'])
@developer_required
def admin_logs():
    runtime = get_runtime_settings()
    configured_log_path = runtime.get('log_file_path') or LOG_FILE_PATH
    log_path = _normalize_dir(configured_log_path) or os.path.abspath(configured_log_path)

    requested_lines = _safe_int(request.args.get('lines', '300'), 300)
    max_lines = min(max(requested_lines, 50), 2000)
    level = (request.args.get('level') or 'ALL').strip().upper()
    query_text = (request.args.get('q') or '').strip()
    query_lower = query_text.lower()

    raw_tail = []
    entries = []
    file_exists = os.path.exists(log_path)

    if file_exists:
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_tail = list(deque(f, maxlen=max_lines))
        except Exception as e:
            flash(f'Unable to read log file: {e}', 'error')

    for line in raw_tail:
        line_text = line.rstrip('\n')
        line_upper = line_text.upper()
        if level != 'ALL' and f' {level} ' not in line_upper:
            continue
        if query_lower and query_lower not in line_text.lower():
            continue
        entries.append(line_text)

    total_errors = sum(1 for line in raw_tail if ' ERROR ' in line.upper())
    total_warnings = sum(1 for line in raw_tail if ' WARNING ' in line.upper())

    return render_template(
        'admin_logs.html',
        log_path=log_path,
        file_exists=file_exists,
        entries=entries,
        lines=max_lines,
        level=level,
        q=query_text,
        total_scanned=len(raw_tail),
        total_matches=len(entries),
        total_errors=total_errors,
        total_warnings=total_warnings,
    )

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
        'inbound_adapter': runtime.get('inbound_adapter'),
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

    # Start IMAP fetcher only when adapter mode includes IMAP.
    inbound_adapter = (runtime.get('inbound_adapter') or INBOUND_ADAPTER).lower()
    if inbound_adapter in {'imap', 'hybrid'}:
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
    else:
        print(f">>> IMAP fetcher skipped due to inbound adapter mode: {inbound_adapter}")

    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', '5000'))
    debug = os.getenv('FLASK_DEBUG', '0') == '1'
    app.run(host=host, port=port, debug=debug)
