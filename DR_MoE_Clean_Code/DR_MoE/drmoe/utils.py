import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def config_load(path):
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2)


def metric_scores(y, pred):
    true = np.asarray(y).ravel()
    forecast = np.asarray(pred).ravel()
    rmse = float(np.sqrt(np.mean((true - forecast) ** 2)))
    mae = float(np.mean(np.abs(true - forecast)))
    valid = np.abs(true) > 1e-6
    mape = float(np.mean(np.abs((true[valid] - forecast[valid]) / true[valid]))) if valid.any() else None
    variance = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1 - np.sum((true - forecast) ** 2) / variance) if variance > 0 else None
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}
