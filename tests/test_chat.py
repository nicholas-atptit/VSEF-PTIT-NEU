import httpx
import asyncio

async def test_chat():
    url = "http://127.0.0.1:8005/api/v2/chat"
    payload = {
        "message": "Tin tức gần đây của mã VGI như thế nào? Có nên đầu tư không?",
        "history": [],
        "ticker": "VGI"
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, timeout=60.0)
            print(f"Status: {resp.status_code}")
            print(f"Response: {resp.json()}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_chat())
