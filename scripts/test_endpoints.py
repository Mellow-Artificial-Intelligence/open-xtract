import json
import sys

import requests

BASE = "http://0.0.0.0:8000"


def main() -> int:
    try:
        # Health
        r = requests.get(f"{BASE}/")
        print("GET / ->", r.status_code, r.text)

        # Extract - document
        payload_doc = {
            "content_url": "https://bitcoin.org/bitcoin.pdf",
            "model": "google/gemini-2.5-flash-lite",
            "schema": {
                "name": "document_summary",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "main_points": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["title", "main_points"],
                    "additionalProperties": False
                }
            },
            "prompt": "Extract the title and main points"
        }
        r = requests.post(f"{BASE}/extract", headers={"Content-Type": "application/json"}, data=json.dumps(payload_doc))
        print("POST /extract (doc) ->", r.status_code)
        print(r.text[:1000])

        # Extract - image
        payload_img = {
            "content_url": "https://fastapi.tiangolo.com/img/index/index-01-swagger-ui-simple.png",
            "model": "google/gemini-2.5-flash-lite",
            "schema": {
                "name": "image_description",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "objects": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["description"],
                    "additionalProperties": False
                }
            },
            "prompt": "Describe the image and list objects"
        }
        r = requests.post(f"{BASE}/extract", headers={"Content-Type": "application/json"}, data=json.dumps(payload_img))
        print("POST /extract (img) ->", r.status_code)
        print(r.text[:1000])

        return 0
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

