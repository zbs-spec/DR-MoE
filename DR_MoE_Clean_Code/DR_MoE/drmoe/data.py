from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from statsmodels.nonparametric.smoothers_lowess import lowess
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, values, starts, input_length, horizon):
        self.values = np.asarray(values, dtype=np.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.input_length = input_length
        self.horizon = horizon

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        s = self.starts[index]
        return self.values[s:s + self.input_length].copy(), self.values[s + self.input_length:s + self.input_length + self.horizon].copy()


def load_series(path, column, date_column=None, aggregate='none'):
    p = Path(path)
    frame = pd.read_csv(p) if p.suffix.lower() != '.parquet' else pd.read_parquet(p)
    if column not in frame:
        raise ValueError(f'Missing column: {column}')
    if date_column:
        if date_column not in frame:
            raise ValueError(f'Missing date column: {date_column}')
        frame[date_column] = pd.to_datetime(frame[date_column])
        frame = frame.sort_values(date_column)
        if aggregate != 'none':
            frame = frame.set_index(date_column)[[column]].resample('D').agg(aggregate).reset_index()
    elif aggregate != 'none':
        raise ValueError('Daily aggregation requires date_column')
    return pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=np.float64)


def interpolate_series(values):
    x = np.arange(len(values))
    good = np.isfinite(values)
    if good.sum() < 2:
        raise ValueError('At least two finite observations are required')
    out = PchipInterpolator(x[good], values[good], extrapolate=True)(x)
    return np.asarray(out, dtype=np.float64)


def make_datasets(raw, input_length, horizon, train_ratio, val_ratio, lowess_frac, smooth=True, train_count=None, val_count=None):
    n = len(raw)
    train_end = int(train_count) if train_count is not None else int(n * train_ratio)
    val_end = train_end + int(val_count) if val_count is not None else int(n * (train_ratio + val_ratio))
    if not 0 < train_end < val_end < n:
        raise ValueError('Invalid split ratios')
    raw = np.asarray(raw, dtype=np.float64)
    segments = [raw[:train_end], raw[train_end:val_end], raw[val_end:]]
    series = np.concatenate([interpolate_series(segment) for segment in segments])
    if smooth:
        if not 0 < lowess_frac <= 1:
            raise ValueError('LOWESS fraction must belong to (0, 1]')
        series = lowess(series[:train_end], np.arange(train_end), frac=lowess_frac, return_sorted=False).tolist() + series[train_end:].tolist()
        series = np.asarray(series, dtype=np.float64)
    mu = float(series[:train_end].mean())
    sd = float(series[:train_end].std())
    if sd == 0:
        raise ValueError('Training standard deviation is zero')
    values = (series - mu) / sd
    starts = np.arange(n - input_length - horizon + 1)
    endpoints = starts + input_length + horizon
    train_starts = starts[endpoints <= train_end]
    val_starts = starts[(starts + input_length >= train_end) & (endpoints <= val_end)]
    test_starts = starts[(starts + input_length >= val_end) & (endpoints <= n)]
    data = [WindowDataset(values, part, input_length, horizon) for part in (train_starts, val_starts, test_starts)]
    if any(len(split) == 0 for split in data):
        raise ValueError('One split is empty. Reduce input length or horizon.')
    return data, {'mean': mu, 'std': sd, 'train_end': train_end, 'val_end': val_end, 'lowess_frac': lowess_frac, 'train_count': train_count, 'val_count': val_count}
