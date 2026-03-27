import asyncio
import httpx
from config.settings import get_settings

async def main():
    settings = get_settings()
    api_key = settings.gemini_api_key.strip()
    
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    models = ["gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-1.5-flash-latest"]
    
    for model in models:
        print(f"\nTesting Model: {model}")
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                data = {"model": model, "messages": [{"role": "user", "content": "OK"}]}
                resp = await client.post(url, headers=headers, json=data, timeout=10.0)
                print(f"Status: {resp.status_code}")
                print(f"Response: {resp.text[:200]}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
