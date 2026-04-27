"""End-to-end smoke test: login to Supabase, then call the backend with the JWT.

Usage:
    python3 scripts/smoke_test.py EMAIL PASSWORD
"""
import json
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError

SUPABASE_URL = "https://hlliwydcsexbbssecajl.supabase.co"
ANON_KEY = "sb_publishable_aLgw0aR9BEYyHkWvF56Ddg_WKIJC0At"
API = "http://localhost:8000"


def http(method, url, headers=None, body=None):
    req = Request(url, method=method, headers=headers or {})
    if body is not None:
        req.data = body.encode()
    try:
        with urlopen(req) as r:
            return r.status, r.read().decode()
    except HTTPError as e:
        return e.code, e.read().decode()


def main():
    if len(sys.argv) != 3:
        print("usage: smoke_test.py EMAIL PASSWORD")
        sys.exit(1)
    email, password = sys.argv[1], sys.argv[2]

    print("1/3  Login to Supabase...")
    status, body = http(
        "POST",
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        {"apikey": ANON_KEY, "Content-Type": "application/json"},
        json.dumps({"email": email, "password": password}),
    )
    if status != 200:
        print(f"     FAILED ({status}): {body}")
        sys.exit(1)
    token = json.loads(body)["access_token"]
    print(f"     OK  token len={len(token)}")

    print("2/5  POST /users/me/sync ...")
    status, body = http(
        "POST",
        f"{API}/users/me/sync",
        {"Authorization": f"Bearer {token}"},
    )
    print(f"     {status}  {body}")

    print("3/5  GET /users/me ...")
    status, body = http(
        "GET",
        f"{API}/users/me",
        {"Authorization": f"Bearer {token}"},
    )
    print(f"     {status}  {body}")

    print("4/5  POST /preferences ...")
    prefs = {
        "budget": "luxury",
        "travel_style": ["adventure", "cultural"],
        "food": ["vegetarian"],
        "pace": "fast",
        "interests": ["hiking", "history"],
        "language": "en",
    }
    status, body = http(
        "POST",
        f"{API}/preferences",
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json.dumps(prefs),
    )
    print(f"     {status}  {body}")

    print("5/5  GET /preferences/me ...")
    status, body = http(
        "GET",
        f"{API}/preferences/me",
        {"Authorization": f"Bearer {token}"},
    )
    print(f"     {status}  {body}")


if __name__ == "__main__":
    main()
