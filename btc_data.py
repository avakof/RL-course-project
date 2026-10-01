"""Download and validate hourly Binance spot BTC/USDT history for PPO.

Official source: https://github.com/binance/binance-public-data
Monthly ZIPs are retained with verified SHA-256 checksums for reproducibility.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import ssl
import time
from urllib.request import urlopen
from zipfile import ZipFile

import certifi
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TRAIN_START = pd.Timestamp('2020-04-01', tz='UTC')
DATA_START = TRAIN_START - pd.Timedelta(days=7)
EVAL_START = pd.Timestamp('2026-08-17', tz='UTC')
EVAL_END = pd.Timestamp('2026-09-17', tz='UTC')
DATA_PATH = ROOT / 'data/binance-BTCUSDT-1h-20200325-20260917.pkl'
LEGACY_PATH = ROOT / 'data/binance-BTCUSDT-1h-20250810-20260917.pkl'
COLUMNS = ['open', 'high', 'low', 'close', 'volume']
ARCHIVES = ROOT / 'data/binance_archives'
BASE_URL = 'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/'


def validate(frame, start=DATA_START, end=EVAL_END, declared_missing=()):
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()[COLUMNS].astype(float)
    expected = pd.date_range(start, end, freq='h', inclusive='left')
    declared = pd.DatetimeIndex(pd.to_datetime(list(declared_missing), utc=True))
    if not frame.index.equals(expected.difference(declared)):
        missing = expected.difference(frame.index)
        raise ValueError(f'Incomplete/duplicate hourly history: {len(missing)} missing; '
                         f'first missing: {list(missing[:5])}')
    prices = frame[['open', 'high', 'low', 'close']]
    if (not np.isfinite(frame.to_numpy()).all() or (prices <= 0).any().any()
            or (frame.volume < 0).any()
            or (frame.high < prices.max(axis=1)).any()
            or (frame.low > prices.min(axis=1)).any()):
        raise ValueError('Invalid OHLCV values')
    return frame


def load_data(path=DATA_PATH):
    """No implicit network access or substitution of the older, shorter cache."""
    if not Path(path).exists():
        raise FileNotFoundError(f'{path} is missing. Run: .venv-rl/bin/python btc_data.py')
    frame = pd.read_pickle(path)
    return validate(frame, declared_missing=frame.attrs.get('missing_hours', ()))


def fetch(url):
    context = ssl.create_default_context(cafile=certifi.where())
    for attempt in range(3):
        try:
            with urlopen(url, context=context, timeout=30) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def parse_archive(payload):
    with ZipFile(BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.endswith('.csv')]
        if len(names) != 1:
            raise ValueError('Expected one candle CSV per archive')
        frame = pd.read_csv(archive.open(names[0]), header=None, usecols=range(6))
    timestamps = frame.pop(0).astype('int64')
    # Binance changed spot archives from milliseconds to microseconds in 2025.
    unit = 'us' if timestamps.iloc[0] >= 10**15 else 'ms'
    frame.columns = COLUMNS
    frame.index = pd.DatetimeIndex(pd.to_datetime(timestamps, unit=unit, utc=True), name='date_open')
    return frame.astype(float)


def download_month(month):
    name = f'BTCUSDT-1h-{month}.zip'
    path = ARCHIVES / name
    checksum_path = ARCHIVES / (name + '.CHECKSUM')
    checksum = checksum_path.read_bytes() if checksum_path.exists() else fetch(BASE_URL + name + '.CHECKSUM')
    payload = path.read_bytes() if path.exists() else fetch(BASE_URL + name)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != checksum.decode().split()[0]:
        raise ValueError(f'Archive checksum mismatch: {name}')
    path.write_bytes(payload)
    checksum_path.write_bytes(checksum)
    print(f'Verified {month}', flush=True)
    return parse_archive(payload), dict(url=BASE_URL + name, sha256=digest)


def download():
    """Backfill through the existing cache start; preserve its evaluation window."""
    legacy = pd.read_pickle(LEGACY_PATH)
    legacy = validate(legacy, pd.Timestamp('2025-08-10', tz='UTC'), EVAL_END)
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(DATA_START.tz_localize(None), legacy.index.min().tz_localize(None), freq='M')
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download_month, map(str, months)))
    downloaded = pd.concat([frame for frame, _ in results]).sort_index()
    # Check the overlap, then retain the original cache from its first candle.
    overlap = downloaded.index.intersection(legacy.index)
    if len(overlap) == 0 or not np.allclose(downloaded.loc[overlap], legacy.loc[overlap], rtol=1e-8, atol=1e-8):
        raise ValueError('Downloaded data does not match the existing cache overlap')
    combined = pd.concat([downloaded.loc[(downloaded.index >= DATA_START)
                                        & (downloaded.index < legacy.index.min())], legacy])
    expected = pd.date_range(DATA_START, EVAL_END, freq='h', inclusive='left')
    missing = expected.difference(combined.index)
    combined.attrs['missing_hours'] = [str(timestamp) for timestamp in missing]
    combined = validate(combined, declared_missing=combined.attrs['missing_hours'])
    temporary = DATA_PATH.with_suffix('.tmp.pkl')
    combined.to_pickle(temporary)
    temporary.replace(DATA_PATH)
    manifest = dict(source='Binance spot BTCUSDT 1h', train_start=str(TRAIN_START),
        start=str(combined.index.min()), end=str(combined.index.max()), rows=len(combined),
        overlap_rows_checked=len(overlap), downloaded_at=str(pd.Timestamp.now(tz='UTC')),
        missing_hours=combined.attrs['missing_hours'],
        gap_policy='No imputation; PPO features restart and training episodes do not cross gaps.',
        archives=[record for _, record in results],
        retained_cache=dict(path=LEGACY_PATH.name, sha256=hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest()),
        sha256=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest())
    DATA_PATH.with_suffix('.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ['start', 'end', 'rows', 'overlap_rows_checked']}, indent=2))


if __name__ == '__main__':
    download()
