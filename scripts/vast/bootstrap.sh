#!/usr/bin/env bash
# Bootstrap a Vast.ai (or any CUDA) instance for the ACNE04-v2 recovery pipeline.
# Usage:
#   REPO_URL=... WORKDIR=/workspace/... OUTPUT_ROOT=/workspace/... bash bootstrap.sh
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Pandeigede2803/acne_segmentation_github.git}"
WORKDIR="${WORKDIR:-/workspace/acne_segmentation_github}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/acne_project/output}"

as_root() {
  if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi
}

echo "== System packages =="
as_root apt-get update -y
as_root apt-get install -y --no-install-recommends git rsync tmux curl ca-certificates

if ! command -v rclone >/dev/null 2>&1; then
  echo "== Install rclone =="
  curl -fsSL https://rclone.org/install.sh | as_root bash
fi

echo "== Repository =="
if [ ! -d "$WORKDIR/.git" ]; then
  as_root mkdir -p "$(dirname "$WORKDIR")"
  git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
echo "Commit: $(git rev-parse HEAD)"

echo "== Python packages =="
python -m pip install --upgrade pip
python -m pip install --quiet opencv-python-headless pandas matplotlib

echo "== Layout =="
as_root mkdir -p "$OUTPUT_ROOT"
as_root mkdir -p /content/drive/MyDrive/acne_project
# Manifests use /content/drive/... absolute paths; keep them valid without rewriting.
as_root ln -sfn "$OUTPUT_ROOT" /content/drive/MyDrive/acne_project/output

echo "== Runtime =="
nvidia-smi || true
python - <<'PY'
import torch, torchvision
print('torch      :', torch.__version__)
print('torchvision:', torchvision.__version__)
print('cuda       :', torch.version.cuda)
print('available  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu        :', torch.cuda.get_device_name(0))
PY

cat > "$OUTPUT_ROOT/run_env.txt" <<EOF
date: $(date -u)
commit: $(git rev-parse HEAD)
host: $(hostname)
EOF
nvidia-smi >> "$OUTPUT_ROOT/run_env.txt" 2>&1 || true
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)" >> "$OUTPUT_ROOT/run_env.txt"
python -m pip freeze >> "$OUTPUT_ROOT/run_env.txt"

echo
echo "Selesai. Contoh sinkronisasi:"
echo "  rclone copy gdrive:acne_project/output/experiments_512_mask_v2/recovery_v1 \\"
echo "    $OUTPUT_ROOT/experiments_512_mask_v2/recovery_v1 -P"
