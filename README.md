\# Camera-Based Health Monitor



A Python desktop app that estimates \*\*heart rate\*\* from a webcam using

remote photoplethysmography (rPPG), shows an \*\*experimental blood pressure

estimate\*\*, and raises an \*\*emergency alert\*\* (alarm + email) when the

heart rate stays abnormal.



> \*\*Disclaimer:\*\* This is an educational demo, not a medical device.

> The blood pressure value is an experimental estimate. In an emergency, call 112.



!\[Screenshot](demo.png)



\## Features

\- Real-time heart rate from the face (forehead and both cheeks)

\- Signal-quality score, so poor readings are rejected

\- Experimental blood pressure estimate, calibrated with one cuff reading

\- Emergency alert: sound, flashing banner and email, triggered by heart rate

&#x20; above 120 or below 45 BPM for 30 seconds

\- Dashboard with a 60-second heart rate trend



\## How it works

1\. \*\*MediaPipe Face Landmarker\*\* finds the face and picks skin regions.

2\. The \*\*average green value\*\* of those regions is recorded every frame

&#x20;  (blood absorbs green light, so pulse shows up as tiny changes).

3\. The signal is resampled, detrended and \*\*band-pass filtered\*\* (0.7-3.0 Hz).

4\. An \*\*FFT\*\* finds the dominant frequency; BPM = frequency x 60.

5\. A quality score and median smoothing keep the display stable.

6\. Blood pressure = a cuff baseline adjusted by changes in heart rate and

&#x20;  pulse shape (placeholder coefficients, not clinically validated).



\## Setup

```

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

```

The `face\_landmarker.task` model file is included. If it is missing, download

it from the MediaPipe Face Landmarker documentation.



\## Run

```

python main.py

```

Keys: \*\*C\*\* calibrate BP (at rest) | \*\*T\*\* test alert | \*\*A\*\* silence alarm |

\*\*S\*\* screenshot | \*\*Q\*\* quit



\## Email alerts (optional)

Use a Gmail account with an App Password, and set these in the terminal

before running (never put them in code):

```

$env:ALERT\_EMAIL\_FROM = "you@gmail.com"

$env:ALERT\_EMAIL\_PASSWORD = "your-16-letter-app-password"

$env:ALERT\_EMAIL\_TO = "recipient@gmail.com"

```



\## Testing

I compared the app's heart rate with \[a smartwatch / manual pulse count]

in \[lighting conditions].



| Trial | App (BPM) | Reference (BPM) | Difference |

|-------|-----------|-----------------|------------|

| 1     | \[ ]       | \[ ]             | \[ ]        |

| 2     | \[ ]       | \[ ]             | \[ ]        |

| 3     | \[ ]       | \[ ]             | \[ ]        |



\## Limitations

\- Needs bright, even light and a still face; motion and dim light cause errors

\- Skin tone, camera quality and video compression affect accuracy

\- Blood pressure cannot be measured reliably from a normal webcam; the value

&#x20; here is a demo estimate, and alerts use heart rate only

\- Built and tested on Windows with one webcam



\## Future work

\- Motion-robust methods (e.g. CHROM or POS algorithms)

\- A model trained on a real PPG/blood pressure dataset

\- SMS alerts, and saving readings to a file or database



\## Tech stack

Python, OpenCV, MediaPipe, NumPy, SciPy

