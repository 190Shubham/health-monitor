import os
import smtplib
import socket

sender = os.environ.get("ALERT_EMAIL_FROM")
pw = (os.environ.get("ALERT_EMAIL_PASSWORD") or "").replace(" ", "")
print("FROM set:", bool(sender),
      "| password length:", len(pw),
      "| TO set:", bool(os.environ.get("ALERT_EMAIL_TO")))

for port in (465, 587):
    try:
        socket.create_connection(("smtp.gmail.com", port), timeout=10).close()
        print(port, "reachable")
    except Exception as e:
        print(port, "NOT reachable:", repr(e))

try:
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as s:
        s.starttls()
        s.login(sender, pw)
        print("587 login OK")
except Exception as e:
    print("587 failed:", repr(e))

try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as s:
        s.login(sender, pw)
        print("465 login OK")
except Exception as e:
    print("465 failed:", repr(e))