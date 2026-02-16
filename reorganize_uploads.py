# reorganize_uploads.py
import os, sqlite3, shutil

DB_PATH = "history.db"
UPLOADS = "uploads"

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT id, filename, saved_path, category FROM uploads")
rows = c.fetchall()

moved = 0
for id_, filename, saved_path, category in rows:
    if not saved_path:
        continue
    # take basename of saved_path
    base = os.path.basename(saved_path.replace('\\','/'))
    cat = (category or 'other')
    cat = "".join(ch for ch in cat if ch.isalnum() or ch in ('_', '-')).lower() or 'other'
    target_dir = os.path.join(UPLOADS, cat)
    os.makedirs(target_dir, exist_ok=True)
    src_candidates = [
        saved_path,
        os.path.join(UPLOADS, base),
        os.path.join(UPLOADS, 'other', base),
    ]
    src = None
    for s in src_candidates:
        if s and os.path.exists(s):
            src = s; break
    if not src:
        continue
    dst = os.path.join(target_dir, base)
    try:
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.move(src, dst)
            # update DB saved_path
            c.execute("UPDATE uploads SET saved_path = ? WHERE id = ?", (os.path.join(target_dir, base), id_))
            moved += 1
    except Exception as e:
        print("Failed to move", src, e)

conn.commit()
conn.close()
print("Reorganization complete. Files moved:", moved)
