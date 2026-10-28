set -euo pipefail
python train.py --config configs/default.yaml --data "$1" --column "${2:-power}" --output "${3:-runs/dkasc}"
