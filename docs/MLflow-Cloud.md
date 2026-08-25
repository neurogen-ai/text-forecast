# MLflow in the Cloud for citation-forecast

This guide sets up a **single-person, ~$10/month MLflow stack** for running
citation-forecast training and evaluation on Modal, while keeping your local
workflow intact.

## Why this stack?

| Component | Choice | Reason |
|---|---|---|
| MLflow server | Cheap VPS (Hetzner CX21 / DigitalOcean $6 droplet) | Always-on, reliable, well under budget |
| Public access | Cloudflare Tunnel | HTTPS on your own domain without opening firewall ports |
| Tracking DB | Postgres on the same VPS | Handles concurrent Modal runs safely; SQLite risks corruption |
| Artifact store | Cloudflare R2 | S3-compatible, no egress fees, cheap storage — good for large checkpoints later |
| Local backup | Download small artifacts into your mutagen → Dropbox folder | Manual or scripted; big model checkpoints stay in R2 |

## What you need before starting

- A domain whose DNS is managed by Cloudflare.
- A Cloudflare account (free).
- A VPS with Ubuntu 24.04 LTS, SSH access, and Docker + Docker Compose installed.
- Roughly $6/month for the VPS.

---

## Step 1 — Create the VPS

Recommended options:

- **Hetzner CX21**: 2 vCPU / 4 GB RAM / ~€5.90/month
- **DigitalOcean Basic droplet**: 1 vCPU / 1 GB RAM / $6/month (upgrade if needed)

During setup:

1. Add your SSH key.
2. Choose **Ubuntu 24.04 LTS**.
3. Note the server IP.

Connect:

```bash
ssh root@<vps-ip>
```

Update the system:

```bash
apt update && apt upgrade -y
```

---

## Step 2 — Install Docker and Docker Compose

On the VPS:

```bash
# Install Docker
apt install -y ca-certificates curl gnupg lsb-release
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list

apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify:

```bash
docker compose version
```

---

## Step 3 — Create the MLflow directory

On the VPS:

```bash
mkdir -p /opt/mlflow-cloud
mkdir -p /opt/mlflow-cloud/mlflow-server
```

You will place three files inside `/opt/mlflow-cloud`:

1. `docker-compose.yml`
2. `mlflow-server/Dockerfile`
3. `.env`

---

## Step 4 — Create the MLflow Dockerfile

Create `/opt/mlflow-cloud/mlflow-server/Dockerfile`:

```dockerfile
FROM python:3.13-slim

# Build deps for psycopg2
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Pin to the same version used by citation-forecast (see pyproject.toml)
ARG MLFLOW_VERSION=3.9.0
RUN pip install --no-cache-dir \
    mlflow==${MLFLOW_VERSION} \
    psycopg2-binary \
    boto3

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
```

> If the version in `pyproject.toml` is not available as a server image, replace
> the `ARG` with the latest MLflow v2.x version that matches the project.

---

## Step 5 — Create the Docker Compose file

Create `/opt/mlflow-cloud/docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: mlflow
    volumes:
      - pgdata:/var/lib/postgresql/data
    networks:
      - mlflow

  mlflow:
    build:
      context: ./mlflow-server
      args:
        MLFLOW_VERSION: ${MLFLOW_VERSION}
    restart: unless-stopped
    environment:
      MLFLOW_TRACKING_URI: postgresql://mlflow:${POSTGRES_PASSWORD}@postgres/mlflow
      MLFLOW_ARTIFACT_ROOT: s3://${R2_BUCKET}/
      MLFLOW_S3_ENDPOINT_URL: ${R2_ENDPOINT_URL}
      AWS_ACCESS_KEY_ID: ${R2_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${R2_SECRET_ACCESS_KEY}
      AWS_REGION: auto
    depends_on:
      - postgres
    networks:
      - mlflow

  cloudflared:
    image: cloudflare/cloudflared:latest
    restart: unless-stopped
    command: tunnel run
    environment:
      TUNNEL_TOKEN: ${TUNNEL_TOKEN}
    networks:
      - mlflow

volumes:
  pgdata:

networks:
  mlflow:
```

---

## Step 6 — Create the environment file

Create `/opt/mlflow-cloud/.env`:

```bash
# MLflow version — match pyproject.toml
MLFLOW_VERSION=3.9.0

# Postgres
POSTGRES_PASSWORD=<long-random-password>

# Cloudflare R2
R2_BUCKET=citef-mlflow-artifacts
R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=<r2-access-key-id>
R2_SECRET_ACCESS_KEY=<r2-secret-access-key>

# Cloudflare Tunnel token (from Zero Trust dashboard)
TUNNEL_TOKEN=<tunnel-token>
```

Protect it:

```bash
chmod 600 /opt/mlflow-cloud/.env
```

---

## Step 7 — Create the Cloudflare R2 bucket

1. Go to the Cloudflare dashboard → **R2**.
2. Click **Create bucket**.
3. Name it `citef-mlflow-artifacts` (or your preferred name).
4. Go to **R2 → Manage R2 API Tokens**.
5. Create a token with **Object Read & Write** permissions.
6. Copy the **Access Key ID** and **Secret Access Key** into `.env`.
7. Note your **Account ID** for the endpoint URL:
   `https://<account-id>.r2.cloudflarestorage.com`.

---

## Step 8 — Create the Cloudflare Tunnel

1. Go to Cloudflare Zero Trust → **Networks → Tunnels**.
2. Click **Create a tunnel** → choose **Cloudflared**.
3. Name it (e.g. `mlflow`).
4. Choose **Docker** as the connector.
5. Copy the **token** (a long string) into `.env` as `TUNNEL_TOKEN`.
6. Add a **Public Hostname**:
   - Subdomain: `mlflow` (or any subdomain you want)
   - Domain: your Cloudflare-managed domain
   - Service type: **HTTP**
   - URL: `http://mlflow:5000`
7. Save.

Because `cloudflared` is on the same Docker network as the MLflow container, it
can reach it by the service name `mlflow` on port 5000.

---

## Step 9 — Start the server

On the VPS:

```bash
cd /opt/mlflow-cloud
docker compose up -d --build
```

Check logs:

```bash
docker compose logs -f mlflow
docker compose logs -f cloudflared
```

Wait ~30 seconds, then test from your local machine:

```bash
curl https://mlflow.yourdomain.com/api/2.0/mlflow/experiments/list
```

You should get a JSON response with an empty experiment list.

Open `https://mlflow.yourdomain.com` in a browser. You should see the MLflow UI.

---

## Step 10 — Configure citation-forecast

In your local repo, edit `config/config.toml`:

```toml
[env]
tracking_uri = "https://mlflow.yourdomain.com"
# raw_loc, staged_loc, artifact_loc stay local
```

The `artifact_loc` in `[env]` is still used for the `LocalCheckpointProcessor`
when running locally. When running on Modal, artifacts go to R2 via the
`MlflowCheckpointProcessor`.

---

## Step 11 — Configure Modal secrets

So Modal containers can log to your MLflow server and upload artifacts to R2,
create these secrets in Modal:

```bash
modal secret create mlflow-creds \
  MLFLOW_TRACKING_URI=https://mlflow.yourdomain.com \
  MLFLOW_TRACKING_USERNAME= \
  MLFLOW_TRACKING_PASSWORD= \
  AWS_ACCESS_KEY_ID=<r2-access-key-id> \
  AWS_SECRET_ACCESS_KEY=<r2-secret-access-key> \
  MLFLOW_S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
```

If you add basic auth to MLflow later, set `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD`.

Reference the secret in `config.toml` under `[runtime.modal]` when plan 2.1
implements secrets management:

```toml
[runtime.modal]
secrets = ["mlflow-creds"]
```

---

## Step 12 — Verify end-to-end

### Local test

Run a local training smoke test:

```bash
citef train -s smoke --no-gpu --subsample 512
```

Check the MLflow UI. You should see a new run under the experiment, and the
checkpoint should appear in your R2 bucket.

### Modal test

Run a Modal training smoke test (after plan 2.0 is implemented):

```bash
citef --experiment graph_embed_class train -s smoke --subsample 512 --runtime modal
```

The spawned job should log metrics to the same MLflow server, and checkpoints
should land in R2.

### Check R2

In the Cloudflare dashboard, go to **R2 → citef-mlflow-artifacts**. You should
see artifact paths like `<experiment-id>/<run-id>/artifacts/...`.

---

## Step 13 — Back up small files locally

For small artifacts (eval exports, small checkpoints), download them into your
mutagen-synced Dropbox folder.

### Option A — MLflow CLI

```bash
mlflow artifacts download \
  -r <run-id> \
  -d ~/Dropbox/citef-mlflow-backups/<run-id>
```

### Option B — rclone sync (recommended for automation)

Install `rclone` locally and configure an R2 remote:

```bash
rclone config
# Choose s3, provider Cloudflare, endpoint from R2, access key, secret key
```

Sync all artifacts down:

```bash
rclone sync r2:citef-mlflow-artifacts ~/Dropbox/citef-mlflow-backups
```

You can run this manually or schedule it with cron.

---

## Large models and future considerations

For multi-billion-parameter models, checkpoints can be tens to hundreds of
gigabytes. R2 remains a good home because:

- Storage is cheap (~$0.015/GB/month).
- There are **no egress fees** from R2.
- It is S3-compatible, so MLflow handles it natively.

However, uploading a 100 GB checkpoint through `mlflow.log_artifact` may be
slow. If that becomes a bottleneck, a future optimization is:

1. Save the latest checkpoint to the Modal `checkpoint_volume` for fast
   in-run resume.
2. Sync it to R2 asynchronously as the durable archive.

For now, with small models, R2 as the canonical artifact store is the simplest
and cheapest path.

---

## Troubleshooting

### `curl` to the MLflow URL fails

- Check that `cloudflared` is running: `docker compose logs cloudflared`
- Check that the public hostname in the Cloudflare dashboard points to
  `http://mlflow:5000` and that the tunnel status shows **Healthy**.
- Check the MLflow container logs: `docker compose logs mlflow`

### Artifacts fail to upload to R2

- Verify `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, and `R2_SECRET_ACCESS_KEY`.
- Ensure the R2 token has **Object Read & Write** permission.
- Check that the bucket name in `MLFLOW_ARTIFACT_ROOT` matches exactly.

### MLflow UI shows runs but no artifacts

- Confirm `MLFLOW_ARTIFACT_ROOT` is set to `s3://citef-mlflow-artifacts/`.
- Check R2 directly in the Cloudflare dashboard to confirm the upload path.

### Database errors

- Ensure the Postgres container is healthy: `docker compose ps`
- Check the password in `.env` matches between `postgres` and `mlflow` services.

---

## Cost estimate

| Item | Approximate monthly cost |
|---|---|
| VPS (Hetzner CX21 or DO $6 droplet) | ~$6 |
| Cloudflare Tunnel + DNS | Free |
| R2 storage for small models | ~$0.05–$0.50 |
| **Total** | **~$6–7/month** |

For large checkpoints, add ~$0.015 per GB stored per month.
