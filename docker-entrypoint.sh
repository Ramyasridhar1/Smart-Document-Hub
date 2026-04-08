#!/bin/sh
set -eu

RUNTIME_ENV="/var/lib/smartdoc/runtime.env"
APP_ENV="/app/.env"
TEMPLATE_ENV="/app/.env.example"

mkdir -p /var/lib/smartdoc

if [ ! -f "$RUNTIME_ENV" ]; then
  if [ -f "$TEMPLATE_ENV" ]; then
    cp "$TEMPLATE_ENV" "$RUNTIME_ENV"
  else
    cat > "$RUNTIME_ENV" <<'EOF'
FLASK_SECRET_KEY=__GENERATE__
SETTINGS_ENCRYPTION_KEY=__GENERATE__
ADMIN_USER=admin
ADMIN_PASS=__GENERATE_ADMIN_PASS__
EOF
  fi
  echo "[entrypoint] Created runtime env file at $RUNTIME_ENV"
fi

python - "$RUNTIME_ENV" <<'PY'
import secrets
import sys

runtime_env_path = sys.argv[1]

with open(runtime_env_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

secret_keys = {'FLASK_SECRET_KEY', 'SETTINGS_ENCRYPTION_KEY', 'SECRET_KEY'}
secret_markers = {'', '__GENERATE__', 'CHANGE_ME', 'replace-me-with-random-hex'}
admin_markers = {'', '__GENERATE_ADMIN_PASS__', 'CHANGE_ME'}

generated = {}
new_lines = []
for line in lines:
    raw = line.rstrip('\n')
    stripped = raw.strip()
    if not stripped or stripped.startswith('#') or '=' not in raw:
        new_lines.append(line)
        continue

    key, value = raw.split('=', 1)
    key = key.strip()
    value = value.strip()

    if key in secret_keys and value in secret_markers:
        value = secrets.token_hex(32)
        generated[key] = value
    elif key == 'ADMIN_PASS' and value in admin_markers:
        value = secrets.token_urlsafe(12)
        generated[key] = value

    new_lines.append(f"{key}={value}\n")

with open(runtime_env_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

if generated:
    print('[entrypoint] Generated startup secrets for:', ', '.join(sorted(generated.keys())))
    if 'ADMIN_PASS' in generated:
        print('[entrypoint] Generated ADMIN_PASS for first login:', generated['ADMIN_PASS'])
PY

cp "$RUNTIME_ENV" "$APP_ENV"

set -a
. "$RUNTIME_ENV"
set +a

exec "$@"
