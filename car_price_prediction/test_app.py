import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
server = subprocess.Popen(
    [sys.executable, "app.py"],
    cwd=BASE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
try:
    time.sleep(2)
    with urllib.request.urlopen("http://127.0.0.1:5000/api/health", timeout=5) as r:
        assert json.loads(r.read())["status"] == "ok"

    payload = json.dumps({
        "Car_Name": "swift",
        "Year": 2017,
        "Present_Price": 6.5,
        "Driven_kms": 30000,
        "Fuel_Type": "Petrol",
        "Selling_type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0,
    }).encode()

    req = urllib.request.Request(
        "http://127.0.0.1:5000/api/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read())
        assert "predicted_price" in data
        assert data["predicted_price"] >= 0

    print("Smoke test passed.")
finally:
    server.terminate()
    server.wait(timeout=5)
