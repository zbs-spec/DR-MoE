import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from drmoe.model import DRMoE
from drmoe.data import load_series, make_datasets
from drmoe.utils import config_load, save_json, set_seed, metric_scores


def evaluate(model, loader, device, mean, std):
    model.eval()
    outputs, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).cpu().numpy()
            outputs.append(pred)
            targets.append(y.numpy())
    pred = np.concatenate(outputs) * std + mean
    true = np.concatenate(targets) * std + mean
    return metric_scores(true, pred), true, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--data', required=True)
    parser.add_argument('--column', default='power')
    parser.add_argument('--date-column', default=None)
    parser.add_argument('--output', default='runs/default')
    parser.add_argument('--ablation', choices=['none', 'global', 'local', 'router'], default='none')
    args = parser.parse_args()
    cfg = config_load(args.config)
    set_seed(cfg['seed'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw = load_series(args.data, args.column, args.date_column, cfg['data']['aggregate'])
    splits, stats = make_datasets(raw, cfg['model']['input_length'], cfg['model']['horizon'], cfg['data']['train_ratio'], cfg['data']['val_ratio'], cfg['data']['lowess_frac'], cfg['data']['smooth'], cfg['data'].get('train_count'), cfg['data'].get('val_count'))
    loaders = [DataLoader(part, batch_size=cfg['training']['batch_size'], shuffle=(i == 0), num_workers=0) for i, part in enumerate(splits)]
    model = DRMoE(**cfg['model'], ablation=args.ablation)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    loss_fn = nn.MSELoss()
    best = float('inf')
    bad = 0
    history = []
    for epoch in range(cfg['training']['epochs']):
        model.train()
        losses = []
        for x, y in loaders[0]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in loaders[1]:
                val_losses.append(loss_fn(model(x.to(device)), y.to(device)).item())
        validation = float(np.mean(val_losses))
        history.append({'epoch': epoch + 1, 'train_mse': float(np.mean(losses)), 'val_mse': validation})
        print(f'Epoch {epoch + 1}: train_mse={np.mean(losses):.6f} val_mse={validation:.6f}', flush=True)
        if validation < best:
            best = validation
            bad = 0
            torch.save({'model': model.state_dict(), 'config': cfg, 'stats': stats, 'ablation': args.ablation}, output / 'best.pt')
        else:
            bad += 1
            if bad >= cfg['training']['patience']:
                break
    checkpoint = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    metrics, true, pred = evaluate(model, loaders[2], device, stats['mean'], stats['std'])
    np.savez(output / 'predictions.npz', true=true, predicted=pred)
    save_json(output / 'metrics.json', metrics)
    save_json(output / 'history.json', history)
    save_json(output / 'run_config.json', {'config': cfg, 'stats': stats, 'data': args.data, 'column': args.column, 'ablation': args.ablation})
    print(metrics)


if __name__ == '__main__':
    main()
