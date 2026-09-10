# forecite in a container on Strix Halo (uv + ROCm torch)

Strix Halo (Ryzen AI Max, `gfx1151`) needs ROCm 7 or newer. Official
`torch==2.5.1` in `pyproject.toml` has no gfx1151 kernels, so this setup
overrides torch to a ROCm 7 build inside the container while the repo stays
as-is for Modal (cloud runs already have their own GPU image).

Plan:

1. Build a Docker image with ROCm runtime + uv.
2. Sync the project venv inside the container with a ROCm 7 torch wheel.
3. Run the train CLI against `/dev/kfd` with the repo mounted.
4. Verify the iGPU is actually used.

## 1. Containerfile

Save as `Containerfile` in the repo root (not committed unless you want it):

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# ROCm 7 userspace only (no GPU inside the build stage); the host kernel
# driver (amdgpu + /dev/kfd) is what actually talks to the iGPU.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Deps first for layer caching: torch ROCm index + project deps
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project \
        --extra rocm || uv sync --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra rocm

# Slim runtime layer: reuse the same venv, drop build caches
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 git ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH" VIRTUAL_ENV=/app/.venv
WORKDIR /app
ENTRYPOINT ["forecite"]
```

If you skip the `--extra rocm` route (section 2) and instead let
`uv sync` resolve torch from the ROCm index directly, drop the `--extra`
from both `RUN uv sync` lines and keep the pyproject index config from
section 2b.

## 2. Getting a gfx1151-capable torch via uv

Two ways; pick one.

### 2a. Quick override, no pyproject change (throwaway verify)

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python \
    torch --index-url https://download.pytorch.org/whl/rocm7.0
uv pip install --python .venv/bin/python -e .
```

### 2b. Proper: platform-conditional index in pyproject.toml

Add to `pyproject.toml` so the lockfile records both CPU (Linux CI, Modal)
and ROCm (this container) torch builds:

```toml
[project.optional-dependencies]
modal = ["modal>=0.73.0"]
test = ["pytest>=8.0"]
rocm = ["torch>=2.9,<2.10"]

[tool.uv]
# ROCm 7 wheels carry gfx1151 (Strix Halo) kernels.
[tool.uv.sources]
torch = [{ index = "pytorch-rocm", marker = "extra == 'rocm'" }]

[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm7.0"
explicit = true
```

Then inside the container: `uv sync --extra rocm`. The default (CPU) torch
stays for every other platform; `--extra rocm` selects the HIP build.

Note: torch 2.5.1 pinned in the base `dependencies` conflicts with the
rocm extra. Either relax the pin (`torch>=2.5.1`) or keep 2.5.1 for the
default env and let the extra replace it — uv resolves per-environment, so
`uv sync` (no extra) and `uv sync --extra rocm` can produce different
lockfile-adjacent installs. If uv complains, relax the pin; Modal accepts
newer torch fine.

## 3. Build and run

```bash
cd ~/projects/forecite
docker build -t forecite:rocm .
```

Run (the flags are the important part; they are what hands the iGPU to the
container):

```bash
docker run --rm -it \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    --ipc=host \
    --security-opt seccomp=unconfined \
    -v ~/projects/forecite:/work \
    -v ~/Data:/data \
    -e HF_HOME=/data/hf-cache \
    -p 5000:5000 \
    forecite:rocm bash
```

- `/dev/kfd` + `/dev/dri` are the AMD compute + DRM nodes. The iGPU is not
  passed through as a device like NVIDIA `--gpus`; ROCm talks to it via kfd.
- `--ipc=host` avoids /dev/shm exhaustion in DataLoader workers.
- `--security-opt seccomp=unconfined` (or a profile that allows ioctls on
  kfd) prevents `HSA_STATUS_ERROR` at init.
- `--group-add video --group-add render`: check your host numeric GIDs with
  `stat -c '%g %n' /dev/kfd` and pass those numbers instead if they differ
  (`--group-add 44 --group-add 992` is typical on Arch).

## 4. Verify the iGPU

Inside the container:

```bash
# 1. ROCm sees the device (userspace query, no kernel driver needed)
python -c "import ctypes; ctypes.CDLL('libhsa-runtime64.so.1')" && \
    /app/.venv/bin/python - <<'EOF'
import torch
print("torch:", torch.__version__, "| hip runtime:", torch.version.hip)
print("cuda/hip available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("arch:", torch.cuda.get_device_capability(0))  # expect (11, 5)
x = torch.randn(4096, 4096, device="cuda")
print("matmul ok:", (x @ x).sum().item() != 0)
EOF
```

Expected: `torch.version.hip` set, available True, capability `(11, 5)`
(gfx1151), and no crash on the matmul.

## 5. Run training

With `~/Data` mounted at `/data` and `config/config.toml` pointing at it
(or `-v` the real path at its usual location so the repo config needs no
edits):

```bash
forecite --experiment retrieval_forecast train \
    -n strix-smoke --gpu --subsample 1024
```

Notes specific to this machine:

- MLflow server: start one inside the container or on the host
  (`mlflow server --backend-store-uri sqlite:///app/mlflow.db --port 5000`)
  and pass `--tracking-uri`.
- First run downloads the ModernBERT embedder only if you run the preprocess
  embed step; set `HF_HOME` somewhere persistent.

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `torch.cuda.is_available()` False | kfd/dri not mounted, or GID wrong: `ls -n /dev/kfd` and match `--group-add`. |
| `HSA_STATUS_ERROR_OUT_OF_RESOURCES` | Missing `--ipc=host`, or another process holds the iGPU. |
| Works as root, fails as user | GID mismatch: pass numeric video/render GIDs. |
| `undefined symbol` / gfxkernel errors on older torch | You got a pre-ROCm7 wheel; the index must be `rocm7.0`, not `rocm6.x`. |
| IOMMU faults in `dmesg` on host | Add `iommu=pt` to kernel cmdline; Strix Halo unified memory is sensitive to IOMMU mode. |
| Container can't write venv into mounted repo | Build deps into the image (as above) and mount the repo at `/work`, or `uv sync` inside the mount on first run. |

## 7. Daily driver

- Reproduce the env on any machine: `uv sync --extra rocm` (container) /
  `uv sync` (CPU, Modal dev).
- Update deps: `uv lock --upgrade-package torch && uv sync --extra rocm`.
- No conda involvement: delete stale `training/`, `utils/` and
  `forecite*` from `~/miniconda3/lib/python3.13/site-packages` so the
  host CLI stops shadowing; then `uv tool install -e ~/projects/forecite`
  for a host-side CLI that uses CPU torch.
