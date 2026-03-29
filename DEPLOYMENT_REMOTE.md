 # Remote Deployment Guide (Windows Dev -> Linux Docker Server)

This project is developed on a Windows laptop and runs only on a remote server with Docker.
No local Docker and no local SQLite installation are required.

## 1) One-time server setup

```bash
# On the server
mkdir -p /home/work/major/smart-document-hub
cd /home/work/major/smart-document-hub
git clone <YOUR_REPO_URL> .
# Ensure .env exists with your real values
nano .env
```

## 2) Deploy updates (every push)

```bash
# On your laptop
git add .
git commit -m "your change"
git push origin <branch>

# On the server
cd /home/work/major/smart-document-hub
git pull origin <branch>
docker compose build
docker compose up -d
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

## 4) Persistence model (SQLite)

- SQLite DB path in container: `/var/lib/smartdoc/history.db`
- Uploads path in container: `/var/lib/smartdoc/uploads`
- Both are persisted in Docker named volume: `sqlite_data`

Data survives container rebuilds/restarts as long as the volume is not removed.

## 5) Backup SQLite before upgrades

```bash
# On the server, create a backup on host filesystem
mkdir -p /home/work/major/smart-document-hub/backups
docker run --rm \
  -v smart-document-hub_sqlite_data:/data \
  -v /home/work/major/smart-document-hub/backups:/backup \
  alpine sh -c 'cp /data/history.db /backup/history_$(date +%Y%m%d_%H%M%S).db'
```

Note: If your compose project name differs, the volume name may differ from `smart-document-hub_sqlite_data`.
Use `docker volume ls | grep sqlite_data` to confirm.

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

Avoid `docker compose down -v` unless you intentionally want to delete SQLite and uploads data.

## 8) Mail account updates from dashboard

- Log in as admin and open `/admin/settings`.
- Update SMTP/IMAP credentials and routing emails from the form.
- Save to apply changes immediately (no container restart needed).

For encrypted password storage consistency, set one stable secret in `.env`:

```env
SETTINGS_ENCRYPTION_KEY=replace-with-a-long-random-value
```

If this is not set, the app falls back to `FLASK_SECRET_KEY` or `SECRET_KEY`.
