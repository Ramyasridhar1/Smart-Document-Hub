import base64
import hashlib
import os
import db_compat as sqlite3
from datetime import datetime
from typing import Dict, Optional

from cryptography.fernet import Fernet, InvalidToken

SENSITIVE_SETTING_KEYS = {
    'email_pass',
    'imap_pass',
}

DEFAULT_SETTING_KEYS = {
    'email_user',
    'email_pass',
    'smtp_server',
    'smtp_port',
    'imap_host',
    'imap_port',
    'imap_user',
    'imap_pass',
    'route_invoice',
    'route_payslip',
    'route_purchase_order',
    'route_minutes',
    'route_resume',
    'from_name',
    'admin_email',
    'imap_poll_seconds',
    'upload_folder',
    'route_dir_invoice',
    'route_dir_payslip',
    'route_dir_purchase_order',
    'route_dir_minutes',
    'route_dir_resume',
    'route_local_enabled',
    'route_email_enabled',
    'log_file_path',
    'log_level',
    'resume_required_skills',
    'resume_preferred_skills',
    'resume_certificate_bonus',
    'resume_project_bonus',
}


def _derive_fernet_key(raw_secret: str) -> bytes:
    digest = hashlib.sha256(raw_secret.encode('utf-8')).digest()
    return base64.urlsafe_b64encode(digest)


def _get_fernet() -> Optional[Fernet]:
    raw_secret = (
        os.getenv('SETTINGS_ENCRYPTION_KEY')
        or os.getenv('FLASK_SECRET_KEY')
        or os.getenv('SECRET_KEY')
    )
    if not raw_secret:
        return None
    return Fernet(_derive_fernet_key(raw_secret))


def ensure_settings_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute(
            '''CREATE TABLE IF NOT EXISTS settings (
                   key TEXT PRIMARY KEY,
                   value TEXT,
                   is_encrypted INTEGER DEFAULT 0,
                   updated_at TEXT
               )'''
        )
        conn.commit()
    finally:
        conn.close()


def encrypt_value(plain_text: str) -> str:
    f = _get_fernet()
    if not f:
        raise RuntimeError('SETTINGS_ENCRYPTION_KEY or FLASK_SECRET_KEY is required to encrypt settings.')
    return f.encrypt((plain_text or '').encode('utf-8')).decode('utf-8')


def decrypt_value(cipher_text: str) -> str:
    f = _get_fernet()
    if not f:
        raise RuntimeError('SETTINGS_ENCRYPTION_KEY or FLASK_SECRET_KEY is required to decrypt settings.')
    try:
        return f.decrypt((cipher_text or '').encode('utf-8')).decode('utf-8')
    except InvalidToken as exc:
        raise RuntimeError('Failed to decrypt stored setting. Check encryption key consistency.') from exc


def get_setting(db_path: str, key: str, default: Optional[str] = None) -> Optional[str]:
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute('SELECT value, is_encrypted FROM settings WHERE key = ?', (key,))
        row = c.fetchone()
    finally:
        conn.close()

    if not row:
        return default

    value, is_encrypted = row
    if not is_encrypted:
        return value

    try:
        return decrypt_value(value)
    except RuntimeError:
        return default


def set_setting(db_path: str, key: str, value: str, encrypt: bool = False) -> None:
    store_value = value or ''
    if encrypt:
        store_value = encrypt_value(store_value)

    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute(
            '''INSERT INTO settings (key, value, is_encrypted, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   value=excluded.value,
                   is_encrypted=excluded.is_encrypted,
                   updated_at=excluded.updated_at''',
            (key, store_value, 1 if encrypt else 0, now),
        )
        conn.commit()
    finally:
        conn.close()


def get_settings_bulk(db_path: str) -> Dict[str, str]:
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute('SELECT key, value, is_encrypted FROM settings')
        rows = c.fetchall()
    finally:
        conn.close()

    out: Dict[str, str] = {}
    for key, value, is_encrypted in rows:
        if is_encrypted:
            try:
                out[key] = decrypt_value(value)
            except RuntimeError:
                continue
        else:
            out[key] = value
    return out


def get_settings_updated_at(db_path: str) -> Optional[str]:
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute('SELECT MAX(updated_at) FROM settings')
        row = c.fetchone()
    finally:
        conn.close()
    return row[0] if row and row[0] else None
