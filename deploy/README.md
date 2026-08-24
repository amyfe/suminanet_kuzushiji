# Deploying Sumina — IONOS VPS runbook

Target: IONOS VPS, Ubuntu 26.04 LTS, CPU-only, Caddy (automatic free TLS) in
front of a systemd-managed Uvicorn process. See [`INSTALL.md`](../INSTALL.md)
§7 and [`app/backend/README.md`](../app/backend/README.md) for the checklist
this runbook implements, and this directory's other files for the actual
artifacts: [`bootstrap.sh`](bootstrap.sh),
[`systemd/sumina-backend.service`](systemd/sumina-backend.service),
[`caddy/Caddyfile`](caddy/Caddyfile), [`requirements-deploy.txt`](requirements-deploy.txt).

Four phases: A (prerequisite, you handle this) → B (local, before every
deploy) → C (VPS, once) → D (VPS, every redeploy after the first).

## Phase A — Prerequisite: get your intended code onto GitHub

The VPS `git clone`s from GitHub, not your local working tree. Before Phase C
can run, whatever state you want live needs to be committed and pushed to
whichever branch/remote the VPS will clone (`bootstrap.sh` defaults to
`main`, `--branch main`, edit if you're deploying from elsewhere). This is
entirely your own call on timing/branching — nothing in this runbook or the
other `deploy/` files touches git.

One thing worth deciding before you push: `config.py`'s `WEBSITE_CHECKPOINT_DIR`
currently points at `checkpoints/suminanet_recognizer/suminanet_best.pt`.
`checkpoints/README.md` (as of this writing) describes that exact path as "a
separate, still-in-progress training run." If that's changed since — confirm
the training run has actually finished before treating it as production.

## Phase B — Local: build the frontend

```bash
cd app/frontend
npm install
npm run build
```

Leave `VITE_API_BASE` unset — this deployment uses the same-origin topology
(Caddy serves both the static build and `/api/*` under one domain), so
relative `/api/...` paths are correct. See
[`app/frontend/README.md`](../app/frontend/README.md#talking-to-the-backend)
if you ever split frontend/backend across different origins instead.

`dist/` is gitignored and gets `scp`'d to the VPS in Phase C — it isn't
pulled by `git clone` there.

## Phase C — VPS: bootstrap + first deploy

0. **Give the VPS read access to the repo, if it's private (recommended — this
   does not require making the repo public).** GitHub's SSH clone URL
   (`git@github.com:...`, what `bootstrap.sh` uses) authenticates via an SSH
   key, not repo visibility — a private repo works exactly the same way, the
   VPS just needs its own key added:
   ```bash
   # on the VPS, as the user who will run the clone:
   ssh-keygen -t ed25519 -C "sumina-vps-deploy" -f ~/.ssh/sumina_deploy -N ""
   cat ~/.ssh/sumina_deploy.pub
   ```
   Copy that public key into GitHub: repo → **Settings → Deploy keys → Add
   deploy key**, read-only (no write access needed to pull). This scopes
   access to just this one repo, unlike adding the key to a personal GitHub
   account. Then either add `~/.ssh/sumina_deploy` to an SSH agent, or point
   git at it directly via `~/.ssh/config`:
   ```
   Host github.com
       IdentityFile ~/.ssh/sumina_deploy
       IdentitiesOnly yes
   ```

1. **SSH into the fresh VPS** and run the bootstrap script:
   ```bash
   git clone --depth 1 git@github.com:amyfe/kuzushiji_transcription_and_translation.git /tmp/sumina-bootstrap
   sudo bash /tmp/sumina-bootstrap/deploy/bootstrap.sh
   ```
   This installs system packages, creates the `sumina` user, configures
   `ufw`, optionally hardens SSH (confirm key-based login in a **second, still
   open** session before answering yes to that prompt), installs Caddy,
   clones the real repo to `/opt/sumina/repo`, creates the venv, and installs
   Python dependencies CPU-only. It does **not** start anything yet.

   Troubleshooting: if the Caddy apt step fails with a signature/GPG error,
   the Cloudsmith-hosted signing key has occasionally needed rotating in the
   past — check <https://caddyserver.com/docs/install> for the current key
   URL if `bootstrap.sh`'s hardcoded one starts failing.

2. **Copy the production checkpoint** — only the one file, not the whole
   (gitignored, multi-GB) `checkpoints/` tree. `checkpoints/` is entirely
   gitignored, so `git clone` never creates the destination directory —
   create it first, or `scp` fails with "No such file or directory":
   ```bash
   ssh user@your-vps mkdir -p /opt/sumina/repo/checkpoints/suminanet_recognizer
   scp checkpoints/suminanet_recognizer/suminanet_best.pt \
       user@your-vps:/opt/sumina/repo/checkpoints/suminanet_recognizer/suminanet_best.pt
   ```
   (Adjust the path to match whatever `config.py`'s `WEBSITE_CHECKPOINT_DIR`
   actually resolves to at deploy time — see the Phase A note above.) Run
   this from your own machine, in the project root — `checkpoints/.../
   suminanet_best.pt` here is the LOCAL source path, the part after the `:`
   is the path on the VPS. The file this creates on the VPS is a completely
   independent copy — your local file is untouched.

3. **Copy the built frontend**:
   ```bash
   scp -r app/frontend/dist user@your-vps:/opt/sumina/repo/app/frontend/dist
   ```

4. **Create `.env` directly on the VPS**, over SSH — never through any other
   channel:
   ```bash
   ssh user@your-vps
   sudo nano /opt/sumina/repo/.env
   ```
   Contents (see [`.env.example`](../.env.example)):
   ```
   OPENROUTER_API_KEY=<real key>
   CORS_ORIGINS=https://your-domain.example
   API_KEY=<a generated secret>
   ```
   Then:
   ```bash
   sudo chown root:root /opt/sumina/repo/.env
   sudo chmod 600 /opt/sumina/repo/.env
   ```
   `API_KEY` is optional but recommended (see
   [`app/backend/README.md`](../app/backend/README.md#configuration) for what
   it does and doesn't protect against) — if you set it, also build the
   frontend in Phase B with `VITE_API_KEY` set to the same value, and re-copy
   `dist/`.

5. **Point DNS** (A/AAAA) at the VPS's public IP, before starting Caddy if
   possible, to avoid unnecessary ACME retry backoff.

6. **Edit the Caddyfile's placeholder domain**:
   ```bash
   sudo nano /etc/caddy/Caddyfile   # replace your-domain.example
   ```

7. **Start the backend and verify readiness — by JSON body, not `systemctl status`**:
   ```bash
   sudo systemctl start sumina-backend
   curl -s localhost:8000/api/health | python3 -m json.tool
   ```
   Look for `"ready": true`. `systemctl status` will report "active" even if
   the model failed to load (see the comment in
   [`sumina-backend.service`](systemd/sumina-backend.service) — a caught
   startup exception leaves the process running but permanently not-ready,
   and nothing about that trips `Restart=on-failure`). If `ready` is `false`,
   check `journalctl -u sumina-backend` and the `error` field in the health
   response before moving on.

8. **Start Caddy and verify over HTTPS**:
   ```bash
   sudo systemctl reload caddy
   journalctl -u caddy -f   # watch for ACME success
   ```
   Then from any machine: `curl https://your-domain.example/api/health` and
   confirm `"ready": true"` over the real public domain, and load the site in
   a browser.

9. **Confirm rate limiting keys off real client IPs, not Caddy's loopback
   address** — this is the one check that specifically validates the
   `--forwarded-allow-ips=127.0.0.1` flag in the systemd unit. From two
   different real networks, send enough requests to
   `POST /api/transcribe` to approach the 10-req/60s limit and confirm the
   limit applies per-client, not globally across all traffic (which is what
   you'd see if the proxy-header trust chain were broken).

## Phase D — VPS: redeploying after the first time

Smaller than Phase C — **never** re-run the SSH-hardening or `ufw enable`
steps from `bootstrap.sh` against a box that's already hardened.

```bash
cd /opt/sumina/repo
sudo -u sumina git pull
sudo -u sumina .venv/bin/pip install "torch>=2.12.1" "torchvision>=0.27.1" \
    --index-url https://download.pytorch.org/whl/cpu
sudo -u sumina .venv/bin/pip install -r deploy/requirements-deploy.txt
```

If the frontend changed: rebuild locally (Phase B) and `scp` a fresh `dist/`.

```bash
sudo systemctl restart sumina-backend
curl -s localhost:8000/api/health | python3 -m json.tool   # re-verify "ready": true
```

Only if `deploy/caddy/Caddyfile` itself changed:
```bash
sudo cp deploy/caddy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

## After your first successful deploy (optional)

```bash
.venv/bin/pip freeze > deploy/requirements-vps-lock.txt
```
Gives you a CPU-only reproducibility snapshot for future redeploys, without
re-resolving `requirements-deploy.txt`'s floor versions fresh each time and
without reintroducing `requirements-lock.txt`'s CUDA bloat. Not required for
the first deploy — a nice-to-have once the box is stable.
