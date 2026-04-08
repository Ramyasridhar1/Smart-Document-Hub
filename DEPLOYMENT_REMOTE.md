 # Remote Deployment Guide (Windows Dev -> Linux Docker Server)

This project is developed on a Windows laptop and runs only on a remote server with Docker.
No local Docker and no local database installation are required.

## 1) One-time server setup

```bash
# On the server
mkdir -p /home/work/major/smart-document-hub
cd /home/work/major/smart-document-hub
git clone <YOUR_REPO_URL> .
```

No manual `.env` file is required.
On first startup, the container creates a persistent runtime env file at:

`/var/lib/smartdoc/runtime.env`

It auto-generates stable values for `FLASK_SECRET_KEY`, `SETTINGS_ENCRYPTION_KEY`, and `ADMIN_PASS` if placeholders are present.

## 2) Deploy updates (every push)

```bash
# On your laptop
git add .
git commit -m "your change"
git push origin <branch>

# On the server
cd /home/work/major/smart-document-hub
git pull origin <branch>
docker compose up -d --build
```

## 3) Runtime checks

```bash
# Container status
docker compose ps

# App logs
docker compose logs -f app

# Health endpoint
curl http://localhost:5000/status
```

## 4) Persistence model

- App data path in container: `/var/lib/smartdoc`
- Uploads path in container: `/var/lib/smartdoc/uploads`
- Runtime env path in container: `/var/lib/smartdoc/runtime.env`
- Postgres data path in container: `/var/lib/postgresql/data`
- Persisted named volumes:
  - `smartdoc_data` (app data, uploads, runtime env)
  - `postgres_data` (Postgres database)

Data survives container rebuilds/restarts as long as the volumes are not removed.

## 5) Backup Postgres before upgrades

```bash
# On the server, create a backup on host filesystem
mkdir -p /home/work/major/smart-document-hub/backups
docker compose exec -T db pg_dump -U smartdoc -d smartdoc > /home/work/major/smart-document-hub/backups/smartdoc_$(date +%Y%m%d_%H%M%S).sql
```

Optional: back up app data volume (uploads + runtime env) too:

```bash
docker run --rm \
  -v smart-document-hub_smartdoc_data:/data \
  -v /home/work/major/smart-document-hub/backups:/backup \
  alpine sh -c 'tar czf /backup/smartdoc_data_$(date +%Y%m%d_%H%M%S).tgz -C /data .'
```

## 6) Rollback quick path

```bash
# On server, rollback to previous commit
cd /home/work/major/smart-document-hub
git log --oneline -n 5
git checkout <previous_commit_sha>
docker compose build
docker compose up -d
```

## 7) Stop and start

```bash
docker compose down
docker compose up -d
```

Avoid `docker compose down -v` unless you intentionally want to delete Postgres data and uploads/runtime env data.

## 8) Mail account updates from dashboard

- Log in as admin and open `/admin/settings`.
- Update SMTP/IMAP credentials and routing emails from the form.
- Save to apply changes immediately (no container restart needed).

For encrypted password storage consistency, keep `/var/lib/smartdoc/runtime.env` persistent.
The container generates and stores a stable `SETTINGS_ENCRYPTION_KEY` there on first run.

## 9) Add a new SMTP/IMAP mailbox (Gmail or Outlook)

Use this when you want to change the sender mailbox (SMTP) and inbox listener mailbox (IMAP).

### 9.1 Where to enter values in the app

1. Log in as admin.
2. Open /admin/settings.
3. Fill these fields:
   - SMTP Email User
   - SMTP Password
   - SMTP Server
   - SMTP Port
   - IMAP Host
   - IMAP Port
   - IMAP User
   - IMAP Password
4. Save settings.

Notes:
- Most providers require SSL/TLS settings implicitly by host+port.
- Password fields should use an App Password where possible, not your normal login password.

### 9.2 Gmail setup

#### A) Prepare the account

1. Turn on 2-Step Verification for the Google account.
2. Create an App Password (Mail).
3. Use that App Password in SMTP Password and IMAP Password.

#### B) Use these server values

- SMTP Server: smtp.gmail.com
- SMTP Port: 587
- IMAP Host: imap.gmail.com
- IMAP Port: 993

#### C) Username format

- SMTP Email User: full Gmail address (example: yourname@gmail.com)
- IMAP User: full Gmail address

### 9.3 Outlook setup (Outlook.com / Microsoft 365)

#### A) Prepare the account

1. If available, enable multi-factor authentication.
2. If your tenant/account supports App Passwords, create one and use it.
3. If App Passwords are not available, use normal password only if your org policy allows basic auth for SMTP/IMAP.

#### B) Use these server values

- SMTP Server: smtp.office365.com
- SMTP Port: 587
- IMAP Host: outlook.office365.com
- IMAP Port: 993

#### C) Username format

- SMTP Email User: full Outlook address (example: yourname@outlook.com or yourname@company.com)
- IMAP User: full Outlook address

### 9.4 Quick test checklist after saving

1. Open /status and confirm SMTP user is shown.
2. Open /imap_status and confirm IMAP thread is alive.
3. Upload one small file and verify:
   - classification completes,
   - routing email is sent,
   - file appears in history.

### 9.5 Troubleshooting by symptom

- Authentication failed:
  - Wrong password, or normal password used instead of App Password.
  - 2FA enabled but App Password not configured.

- Connection timeout:
  - Wrong host/port, firewall, or provider blocking sign-in.

- Works for SMTP but IMAP fails:
  - IMAP not enabled for the mailbox policy/account.
  - IMAP username/password mismatch.

### 9.6 Optional runtime env equivalents

You can also set the same values in `/var/lib/smartdoc/runtime.env` (or use dashboard settings):

- EMAIL_USER
- EMAIL_PASS
- SMTP_SERVER
- SMTP_PORT
- IMAP_HOST
- IMAP_PORT
- IMAP_USER
- IMAP_PASS
