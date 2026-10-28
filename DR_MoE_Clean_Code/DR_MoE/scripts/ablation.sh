set -euo pipefail
for variant in none global local router; do
  python train.py --config configs/default.yaml --data "$1" --column "${2:-power}" --ablation "$variant" --output "runs/ablation_${variant}"
done
