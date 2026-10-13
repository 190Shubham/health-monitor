import cv2
import numpy as np

PANEL_W = 340
PANEL_H = 480

BG = (30, 24, 20)        # colors are BGR
CARD = (48, 40, 34)
WHITE = (240, 240, 240)
GREY = (165, 165, 165)
GREEN = (120, 220, 120)
ORANGE = (0, 165, 255)
RED = (70, 70, 240)
YELLOW = (0, 220, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _width(s, scale, thick):
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def _card(img, y1, y2):
    cv2.rectangle(img, (10, y1), (PANEL_W - 10, y2), CARD, -1)


def draw_dashboard(face, bpm, quality, quality_min, bp_est, baseline,
                   hr_history, alerts, t):
    """Return a PANEL_H x PANEL_W image showing all readings."""
    p = np.full((PANEL_H, PANEL_W, 3), BG, np.uint8)

    # ---- title and status ----
    _text(p, "HEALTH MONITOR", (20, 28), 0.7, WHITE, 2)
    if alerts.active:
        status, scol = "EMERGENCY", RED
    elif not face:
        status, scol = "NO FACE DETECTED", RED
    elif bpm is None:
        status, scol = "MEASURING...", YELLOW
    else:
        status, scol = "MONITORING", GREEN
    cv2.circle(p, (24, 45), 5, scol, -1, cv2.LINE_AA)
    _text(p, status, (36, 50), 0.5, scol, 1)

    # ---- heart rate card ----
    _card(p, 60, 165)
    _text(p, "HEART RATE", (20, 80), 0.45, GREY)
    if bpm is None:
        hcol = GREY
    elif bpm > alerts.high_bpm or bpm < alerts.low_bpm:
        hcol = RED
    else:
        hcol = GREEN
    num = "--" if bpm is None else str(int(round(bpm)))
    _text(p, num, (20, 148), 2.4, hcol, 4)
    _text(p, "BPM", (20 + _width(num, 2.4, 4) + 12, 148), 0.8, GREY, 1)

    # ---- blood pressure card ----
    _card(p, 175, 240)
    _text(p, "BLOOD PRESSURE (EXPERIMENTAL)", (20, 195), 0.4, GREY)
    if bp_est:
        val = f"{bp_est[0]:.0f}/{bp_est[1]:.0f}"
        _text(p, val, (20, 228), 1.1, ORANGE, 2)
        _text(p, "mmHg", (20 + _width(val, 1.1, 2) + 10, 228), 0.5, GREY, 1)
    elif baseline is None:
        _text(p, "Press C at rest to calibrate", (20, 225), 0.5, YELLOW, 1)
    else:
        _text(p, "Waiting for a clean signal...", (20, 225), 0.5, GREY, 1)

    # ---- heart rate trend (last 60 seconds) ----
    _card(p, 250, 360)
    _text(p, "HEART RATE TREND (60 s)", (20, 268), 0.4, GREY)
    gx1, gx2, gy0, gy1 = 20, 320, 276, 352

    def ymap(v):
        v = min(max(v, 40), 160)
        return int(gy1 - (v - 40) / 120.0 * (gy1 - gy0))

    cv2.rectangle(p, (gx1, gy0), (gx2, gy1), (38, 32, 27), -1)
    for limit in (alerts.low_bpm, alerts.high_bpm):
        y = ymap(limit)
        cv2.line(p, (gx1, y), (gx2, y), (70, 70, 150), 1)
        _text(p, str(limit), (gx1 + 3, y - 3), 0.35, GREY, 1)

    prev = None
    for ts, v in hr_history:
        if v is None or ts < t - 60:
            prev = None
            continue
        pt = (int(gx1 + (ts - (t - 60)) / 60.0 * (gx2 - gx1)), ymap(v))
        if prev is not None:
            cv2.line(p, prev, pt, GREEN, 2, cv2.LINE_AA)
        prev = pt

    # ---- signal quality bar ----
    _text(p, "SIGNAL QUALITY", (20, 382), 0.45, GREY)
    bx1, bx2, by1, by2 = 170, 320, 374, 386
    cv2.rectangle(p, (bx1, by1), (bx2, by2), CARD, -1)
    frac = min(max(quality, 0.0) / 0.6, 1.0)
    qcol = GREEN if quality >= quality_min else ORANGE
    cv2.rectangle(p, (bx1, by1), (bx1 + int(frac * (bx2 - bx1)), by2), qcol, -1)
    mark = bx1 + int(min(quality_min / 0.6, 1.0) * (bx2 - bx1))
    cv2.line(p, (mark, by1 - 3), (mark, by2 + 3), WHITE, 1)

    # ---- alert box ----
    ay1, ay2 = 398, 444
    secs = alerts.abnormal_secs(t)
    if alerts.active:
        flash = int(t * 2) % 2 == 0
        cv2.rectangle(p, (10, ay1), (PANEL_W - 10, ay2),
                      RED if flash else (30, 30, 120), -1)
        _text(p, alerts.reason[:34], (18, 414), 0.5, WHITE, 1)
        _text(p, "Press A to silence", (18, 428), 0.4, WHITE, 1)
    else:
        cv2.rectangle(p, (10, ay1), (PANEL_W - 10, ay2), CARD, -1)
        if secs > 0:
            _text(p, f"Abnormal HR {secs:.0f}/{alerts.hold_sec}s",
                  (18, 414), 0.5, ORANGE, 1)
        else:
            _text(p, "No alerts", (18, 414), 0.5, GREEN, 1)
    if alerts.email_status:
        _text(p, alerts.email_status[:40], (18, 440), 0.4, WHITE if alerts.active else GREY, 1)

    # ---- footer ----
    _text(p, "C cal  T test  A mute  S snap  Q quit", (12, 462), 0.4, GREY, 1)
    _text(p, "Demo only - not a medical device", (12, 476), 0.4, GREY, 1)
    return p