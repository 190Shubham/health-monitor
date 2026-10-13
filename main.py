import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, detrend
from bp import pulse_features, estimate_bp
from alert import AlertSystem
from dashboard import draw_dashboard, PANEL_W, PANEL_H

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)

# ---------- settings you can tune ----------
WINDOW_SEC = 10        # seconds of signal kept
MIN_SEC = 6            # need at least this much before measuring
LOW_HZ, HIGH_HZ = 0.7, 3.0   # 42 - 180 BPM
TARGET_FPS = 30        # resample rate
QUALITY_MIN = 0.25     # reject readings below this quality
CUFF_SBP, CUFF_DBP = 120, 80   # put your real cuff reading here
# -------------------------------------------


def point(lm, i, w, h):
    return int(lm[i].x * w), int(lm[i].y * h)


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    return (max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2))


def get_rois(lm, w, h):
    x1, _ = point(lm, 107, w, h)
    x2, _ = point(lm, 336, w, h)
    _, y_top = point(lm, 10, w, h)
    _, y_brow = point(lm, 9, w, h)
    span = y_brow - y_top
    forehead = (x1, y_top + int(0.15 * span), x2, y_brow - int(0.15 * span))

    xl, _ = point(lm, 234, w, h)
    xr, _ = point(lm, 454, w, h)
    half = int(0.07 * abs(xr - xl))

    cx1, cy1 = point(lm, 50, w, h)
    cx2, cy2 = point(lm, 280, w, h)
    left_cheek = (cx1 - half, cy1 - half, cx1 + half, cy1 + half)
    right_cheek = (cx2 - half, cy2 - half, cx2 + half, cy2 + half)

    return [clip_box(b, w, h) for b in (forehead, left_cheek, right_cheek)]


def roi_green_mean(frame, box):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    return float(frame[y1:y2, x1:x2, 1].mean())


def estimate_bpm(signal):
    """Return (bpm, quality, filtered_signal). bpm is None if not enough data."""
    t = np.array([s[0] for s in signal])
    v = np.array([s[1] for s in signal])
    duration = t[-1] - t[0]
    if duration < MIN_SEC:
        return None, 0.0, None

    # 1) resample to an even sampling rate
    n = int(duration * TARGET_FPS)
    t_even = np.linspace(t[0], t[-1], n)
    v_even = np.interp(t_even, t, v)

    # 2) remove drift, then keep only heartbeat frequencies
    v_even = detrend(v_even)
    nyq = TARGET_FPS / 2
    b, a = butter(3, [LOW_HZ / nyq, HIGH_HZ / nyq], btype="band")
    filtered = filtfilt(b, a, v_even)

    # 3) FFT (Hanning window + zero padding for a smoother peak)
    nfft = 8192
    windowed = filtered * np.hanning(len(filtered))
    power = np.abs(np.fft.rfft(windowed, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / TARGET_FPS)

    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    f_band, p_band = freqs[band], power[band]
    peak = int(np.argmax(p_band))
    bpm = f_band[peak] * 60.0

    # 4) quality = share of power close to the peak
    near = np.abs(f_band - f_band[peak]) <= 0.1
    quality = float(p_band[near].sum() / p_band.sum())

    return bpm, quality, filtered


def draw_signal(frame, values, w, h):
    if values is None or len(values) < 2:
        return
    arr = np.array(values, dtype=np.float32)
    arr -= arr.mean()
    rng = arr.max() - arr.min()
    if rng < 1e-6:
        return
    arr /= rng
    strip_h = 80
    y_mid = h - strip_h // 2 - 10
    xs = np.linspace(10, w - 10, len(arr)).astype(np.int32)
    ys = (y_mid - arr * strip_h).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], False, (0, 255, 255), 2)


signal = deque()
bpm_history = deque(maxlen=8)
hr_history = deque(maxlen=200)   # (time, bpm or None) for the trend graph
display_bpm = None
quality = 0.0
filtered_view = None
last_calc = 0.0
last_hist = 0.0
baseline = None      # set when you press C
rise_ratio = None
bp_est = None
alerts = AlertSystem(high_bpm=120, low_bpm=45, hold_sec=30)

cap = cv2.VideoCapture(0)
start = time.time()

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        t = time.time() - start
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(t * 1000))
        face = bool(result.face_landmarks)

        if face:
            lm = result.face_landmarks[0]
            rois = get_rois(lm, w, h)

            # measure first, before drawing on the frame
            vals = [roi_green_mean(frame, b) for b in rois]
            vals = [v for v in vals if v is not None]
            if vals:
                signal.append((t, sum(vals) / len(vals)))
            while signal and t - signal[0][0] > WINDOW_SEC:
                signal.popleft()

            # recompute heart rate twice per second
            if t - last_calc > 0.5 and len(signal) > 1:
                last_calc = t
                bpm, quality, filtered_view = estimate_bpm(signal)
                if bpm is not None and quality >= QUALITY_MIN:
                    bpm_history.append(bpm)
                elif bpm_history:
                    bpm_history.popleft()  # let old readings fade out
                display_bpm = (float(np.median(bpm_history))
                               if bpm_history else None)

                # experimental blood pressure estimate
                rise_ratio = None
                if filtered_view is not None and quality >= QUALITY_MIN:
                    rise_ratio = pulse_features(filtered_view, TARGET_FPS)
                bp_est = None
                if baseline and display_bpm and rise_ratio:
                    bp_est = estimate_bp(display_bpm, rise_ratio, baseline)

            alerts.update(display_bpm, t)

            for (x1, y1, x2, y2), color in zip(
                    rois, [(0, 255, 0), (255, 150, 0), (255, 150, 0)]):
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            draw_signal(frame,
                        filtered_view if filtered_view is not None
                        else [v for _, v in signal], w, h)
        else:
            signal.clear()
            bpm_history.clear()
            display_bpm = None
            filtered_view = None
            rise_ratio = None
            bp_est = None
            alerts.update(None, t)

        # keep a short history for the trend graph
        if t - last_hist > 0.5:
            last_hist = t
            hr_history.append((t, display_bpm))

        # red border on the video while an alarm is active
        if alerts.active and int(t * 2) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)

        # build the dashboard and put it next to the video
        panel = draw_dashboard(face, display_bpm, quality if face else 0.0,
                               QUALITY_MIN, bp_est, baseline, hr_history,
                               alerts, t)
        panel = cv2.resize(panel, (int(PANEL_W * h / PANEL_H), h))
        output = np.hstack([frame, panel])
        cv2.imshow("Health Monitor", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c") and display_bpm and rise_ratio:
            baseline = (CUFF_SBP, CUFF_DBP, display_bpm, rise_ratio)
        if key == ord("a"):
            alerts.acknowledge()
        if key == ord("t"):
            alerts.trigger("TEST ALERT (key T pressed)", t, test=True)
        if key == ord("s"):
            name = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(name, output)
            print("Saved", name)

cap.release()
cv2.destroyAllWindows()