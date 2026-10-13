import os
import smtplib
import threading
import time
from email.message import EmailMessage

try:
    import winsound  # Windows only
except ImportError:
    winsound = None


class AlertSystem:
    def __init__(self, high_bpm=120, low_bpm=45, hold_sec=30, cooldown_sec=300):
        self.high_bpm = high_bpm
        self.low_bpm = low_bpm
        self.hold_sec = hold_sec
        self.cooldown_sec = cooldown_sec

        self.abnormal_since = None
        self.active = False
        self.is_test = False
        self.reason = ""
        self.email_status = ""
        self.last_email = None

        self._stop = threading.Event()
        self._alarm_thread = None

    # ---------- called every frame ----------
    def update(self, bpm, now):
        if bpm is None:
            # no reliable reading: restart the timer, leave any active alarm alone
            self.abnormal_since = None
            return

        if bpm > self.high_bpm:
            reason = f"High heart rate: {bpm:.0f} BPM"
        elif bpm < self.low_bpm:
            reason = f"Low heart rate: {bpm:.0f} BPM"
        else:
            reason = None

        if reason is None:
            self.abnormal_since = None
            if self.active and not self.is_test:
                self.clear()
            return

        if self.abnormal_since is None:
            self.abnormal_since = now
        if not self.active and now - self.abnormal_since >= self.hold_sec:
            self.trigger(reason, now)

    def abnormal_secs(self, now):
        return 0.0 if self.abnormal_since is None else now - self.abnormal_since

    # ---------- trigger / clear ----------
    def trigger(self, reason, now, test=False):
        self.active = True
        self.is_test = test
        self.reason = reason
        self._start_alarm()

        due = (self.last_email is None
               or now - self.last_email >= self.cooldown_sec)
        if test or due:
            self.last_email = now
            self.email_status = "Sending email..."
            threading.Thread(target=self._send_email,
                             args=(reason, test), daemon=True).start()

    def clear(self):
        self.active = False
        self.is_test = False
        self._stop.set()

    def acknowledge(self):
        """User pressed the silence key."""
        self.clear()
        self.abnormal_since = None

    # ---------- alarm sound ----------
    def _start_alarm(self):
        self._stop.clear()
        if self._alarm_thread and self._alarm_thread.is_alive():
            return
        self._alarm_thread = threading.Thread(target=self._alarm_loop,
                                              daemon=True)
        self._alarm_thread.start()

    def _alarm_loop(self):
        while not self._stop.is_set():
            if winsound:
                winsound.Beep(1000, 400)
                winsound.Beep(1500, 400)
            else:
                print("\a", end="", flush=True)
                time.sleep(0.8)

    # ---------- email ----------
    def _send_email(self, reason, test):
        sender = os.environ.get("ALERT_EMAIL_FROM")
        password = os.environ.get("ALERT_EMAIL_PASSWORD")
        to = os.environ.get("ALERT_EMAIL_TO")
        if not (sender and password and to):
            self.email_status = "Email not set up"
            return

        msg = EmailMessage()
        msg["Subject"] = "TEST - Health Monitor" if test else "ALERT - Health Monitor"
        msg["From"] = sender
        msg["To"] = to
        msg.set_content(
            f"{reason}\n"
            f"Time: {time.strftime('%d-%m-%Y %H:%M:%S')}\n\n"
            "Automatic message from a demo project. This is not a medical "
            "device. In a real emergency, call 112."
        )
        pw = password.replace(" ", "")
        try:
            try:
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as s:
                    s.login(sender, pw)
                    s.send_message(msg)
            except (smtplib.SMTPServerDisconnected, OSError):
                # port 465 blocked or dropped: try port 587 instead
                with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as s:
                    s.starttls()
                    s.login(sender, pw)
                    s.send_message(msg)
            self.email_status = "Email sent"
        except Exception as e:
            self.email_status = f"Email failed: {type(e).__name__}"