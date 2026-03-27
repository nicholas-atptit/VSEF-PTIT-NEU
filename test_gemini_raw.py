import asyncio
import httpx
from config.settings import get_settings

async def main():
    settings = get_settings()
    api_key = settings.gemini_api_key.strip()
    
    # Try the most likely raw endpoints
    endpoints = [
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    ]
    
    for url in endpoints:
        print(f"\nTesting: {url}")
        try:
            async with httpx.AsyncClient() as client:
                if "openai" in url:
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    data = {"model": "gemini-1.5-flash", "messages": [{"role": "user", "content": "OK"}]}
                else:
                    headers = {"Content-Type": "application/json"}
                    data = {"contents": [{"parts": [{"text": "OK"}]}]}
                
                resp = await client.post(url, headers=headers, json=data, timeout=10.0)
                print(f"Status: {resp.status_code}")
                print(f"Response: {resp.text[:200]}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
