import json
import requests

# GET /
r = requests.get("http://127.0.0.1:8000/")
print("GET / status:", r.status_code)
try:
    print("GET / result:", r.json())
except Exception:
    print("GET / result (non-JSON):", r.text)

# Payload for POST /data/
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# POST /data/
r = requests.post(
    "http://127.0.0.1:8000/data/",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data),
)
print("POST /data/ status:", r.status_code)
try:
    print("POST /data/ result:", r.json())
except Exception:
    print("POST /data/ result (non-JSON):", r.text)
