import numpy as np
from scipy.signal import find_peaks


def pulse_features(filtered, fps):
    """Return the 'rise ratio': how long the pulse takes to rise,
    divided by the beat length. None if the signal is too weak."""
    x = np.asarray(filtered, dtype=float)
    peaks, _ = find_peaks(x, distance=int(0.4 * fps),
                          prominence=0.3 * np.std(x))
    troughs, _ = find_peaks(-x, distance=int(0.4 * fps))
    if len(peaks) < 4 or len(troughs) < 4:
        return None

    rises = []
    for p in peaks[1:]:
        before = troughs[troughs < p]
        if len(before) == 0:
            continue
        rises.append((p - before[-1]) / fps)
    if not rises:
        return None

    period = np.median(np.diff(peaks)) / fps
    return float(np.median(rises) / period)


def estimate_bp(hr, rise_ratio, baseline):
    """Rough estimate = cuff baseline + small shifts from changes in
    heart rate and pulse shape. Placeholder coefficients, NOT clinical."""
    sbp0, dbp0, hr0, rise0 = baseline
    d_hr = hr - hr0
    d_rise = rise_ratio - rise0

    sbp = sbp0 + 0.30 * d_hr - 40.0 * d_rise
    dbp = dbp0 + 0.15 * d_hr - 20.0 * d_rise

    sbp = float(np.clip(sbp, sbp0 - 15, sbp0 + 15))
    dbp = float(np.clip(dbp, dbp0 - 10, dbp0 + 10))
    return sbp, dbp
