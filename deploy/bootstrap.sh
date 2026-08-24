#!/usr/bin/env bash
# Sumina — VPS bootstrap (Ubuntu 26.04 LTS, CPU-only)
#
# One-time setup for a fresh IONOS VPS. Re-running is mostly safe (apt/adduser
# steps are idempotent) EXCEPT the SSH-hardening phase, marked below — do not
# blindly re-run this whole script without reading that section again.
#
# Run as root (or via sudo) on the VPS itself. See deploy/README.md Phase C
# for the full sequence this fits into (checkpoint/dist transfer, .env
# creation, DNS, first start — none of that happens in this script).
#
# Usage: sudo bash deploy/bootstrap.sh

set -euo pipefail

REPO_URL="git@github.com:amyfe/kuzushiji_transcription_and_translation.git"
REPO_DIR="/opt/sumina/repo"
APP_USER="sumina"

echo "=== 1. System update ==="
apt update && apt upgrade -y

echo "=== 2. Base packages ==="
# git/curl/ufw/unattended-upgrades/build-essential/python3-venv: bootstrap needs.
# mecab/libmecab-dev/mecab-ipadic-utf8: straight from INSTALL.md section 1.
apt install -y git curl ufw unattended-upgrades build-essential \
    python3 python3-venv python3-pip \
    mecab libmecab-dev mecab-ipadic-utf8

echo "=== 3. Dedicated non-root system user ==="
if ! id -u "$APP_USER" >/dev/null 2>&1; then
    adduser --system --group --no-create-home --shell /usr/sbin/nologin "$APP_USER"
fi

echo "=== 4. Firewall (22/80/443 only) ==="
ufw allow OpenSSH
ufw allow 80
ufw allow 443
ufw --force enable

echo "=== 5. SSH hardening ==="
echo "    !! Before this step commits, confirm key-based SSH login works in a"
echo "    !! SECOND, STILL-OPEN session. If this is a re-run and you already"
echo "    !! hardened SSH, this step is safe to skip (Ctrl+C now)."
read -r -p "    Continue with SSH hardening? [y/N] " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    cat > /etc/ssh/sshd_config.d/99-sumina-hardening.conf <<'EOF'
PermitRootLogin no
PasswordAuthentication no
EOF
    systemctl reload ssh
    echo "    SSH hardening applied."
else
    echo "    Skipped SSH hardening."
fi

echo "=== 6. Automatic security updates ==="
dpkg-reconfigure -f noninteractive unattended-upgrades

echo "=== 7. Caddy (official apt repo — gives caddy.service CAP_NET_BIND_SERVICE) ==="
if ! command -v caddy >/dev/null 2>&1; then
    apt install -y debian-keyring debian-archive-keyring apt-transport-https
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        | tee /etc/apt/sources.list.d/caddy-stable.list
    apt update
    apt install -y caddy
fi

echo "=== 8. Application code ==="
mkdir -p "$(dirname "$REPO_DIR")"
if [[ -d "$REPO_DIR/.git" ]]; then
    git -C "$REPO_DIR" pull
else
    git clone --branch main --depth 1 "$REPO_URL" "$REPO_DIR"
fi
chown -R "$APP_USER":"$APP_USER" "$REPO_DIR"

echo "=== 9. Python venv + CPU-only dependencies ==="
sudo -u "$APP_USER" python3 -m venv "$REPO_DIR/.venv"
sudo -u "$APP_USER" "$REPO_DIR/.venv/bin/pip" install --upgrade pip
# CPU wheel index FIRST — requirements-deploy.txt alone does not prevent pip
# from resolving GPU/CUDA wheels for torch (see that file's own header).
sudo -u "$APP_USER" "$REPO_DIR/.venv/bin/pip" install \
    "torch>=2.12.1" "torchvision>=0.27.1" \
    --index-url https://download.pytorch.org/whl/cpu
sudo -u "$APP_USER" "$REPO_DIR/.venv/bin/pip" install \
    -r "$REPO_DIR/deploy/requirements-deploy.txt"

echo "=== 10. UniDic Edo dictionary (run as sumina, not root) ==="
sudo -u "$APP_USER" "$REPO_DIR/.venv/bin/python" -c "import unidic; unidic.download()" \
    || echo "    UniDic download failed/skipped — falls back to unidic-lite (already installed), see INSTALL.md section 4."

echo "=== 11. systemd unit + Caddyfile ==="
cp "$REPO_DIR/deploy/systemd/sumina-backend.service" /etc/systemd/system/
cp "$REPO_DIR/deploy/caddy/Caddyfile" /etc/caddy/Caddyfile
systemctl daemon-reload
systemctl enable sumina-backend

cat <<'EOF'

=== Bootstrap done. NOT started yet. ===
Still needed before the first start (see deploy/README.md Phase C):
  - scp the production checkpoint to the WEBSITE_CHECKPOINT_DIR path
  - scp the locally-built app/frontend/dist/
  - create /opt/sumina/repo/.env (chmod 600, chown root:root)
  - edit the Caddyfile's your-domain.example placeholder, point DNS at this VPS
  - systemctl start sumina-backend, then verify /api/health's "ready" field
  - systemctl reload caddy
EOF
