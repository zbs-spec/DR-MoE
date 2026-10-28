import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from drmoe.model import DRMoE
from drmoe.data import load_series, make_datasets
from drmoe.utils import metric_scores, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--column', default='power')
    p.add_argument('--date-column', default=None)
    p.add_argument('--output', default='runs/evaluation')
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']
    stats = checkpoint['stats']
    raw = load_series(args.data, args.column, args.date_column, cfg['data']['aggregate'])
    datasets, new_stats = make_datasets(raw, cfg['model']['input_length'], cfg['model']['horizon'], cfg['data']['train_ratio'], cfg['data']['val_ratio'], cfg['data']['lowess_frac'], cfg['data']['smooth'], cfg['data'].get('train_count'), cfg['data'].get('val_count'))
    if abs(new_stats['mean'] - stats['mean']) > 1e-6 or abs(new_stats['std'] - stats['std']) > 1e-6:
        raise ValueError('Dataset statistics differ from those of the checkpoint')
    loader = DataLoader(datasets[2], batch_size=cfg['training']['batch_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DRMoE(**cfg['model'], ablation=checkpoint['ablation']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    y_true, y_pred, weights = [], [], []
    with torch.no_grad():
        for x, y in loader:
            out, extra = model(x.to(device), return_experts=True)
            y_true.append(y.numpy())
            y_pred.append(out.cpu().numpy())
            weights.append(extra['weights'].cpu().numpy())
    y_true = np.concatenate(y_true) * stats['std'] + stats['mean']
    y_pred = np.concatenate(y_pred) * stats['std'] + stats['mean']
    weights = np.concatenate(weights)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / 'evaluation.npz', true=y_true, predicted=y_pred, router_weights=weights)
    metrics = metric_scores(y_true, y_pred)
    save_json(output / 'metrics.json', metrics)
    print(metrics)


if __name__ == '__main__':
    main()
