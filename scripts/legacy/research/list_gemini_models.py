"""Legacy module.
Retained for historical compatibility or migration reference.
Not part of canonical governed runtime.
"""

import requests
from config.settings import get_settings

def main():
    settings = get_settings()
    api_key = settings.gemini_api_key.strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            print("Available Models:")
            for m in models:
                print(f"- {m['name']}")
        else:
            print(f"Failed to list models: {resp.status_code}")
            print(resp.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
