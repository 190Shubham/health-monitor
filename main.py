
import os
from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "project": "Health Monitor",
        "message": "Health Monitor API is running"
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


def run_desktop_monitor():
    """Run the existing OpenCV webcam application locally."""
    import time
    from collections import deque

    import cv2
    import numpy as np
    import mediapipe as mp
    from scipy.signal import butter, filtfilt, detrend

    from bp import pulse_features, estimate_bp
    from alert import AlertSystem
    from dashboard import draw_dashboard, PANEL_W, PANEL_H

    # Move your existing webcam-monitoring code here.
    # Start with your BaseOptions, FaceLandmarker, and settings.
    # Keep the rest of your original processing loop unchanged.

    raise NotImplementedError(
        "Paste your existing webcam-monitoring code into this function."
    )


if __name__ == "__main__":
    if os.environ.get("RUN_DESKTOP") == "1":
        run_desktop_monitor()
    else:
        app.run(
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 5000)),
            debug=False
        )
