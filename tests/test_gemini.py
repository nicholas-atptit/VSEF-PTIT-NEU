import asyncio
from openai import AsyncOpenAI
from config.settings import get_settings

async def main():
    settings = get_settings()
    # Try the most common OpenAI-compatible endpoint for Gemini
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    model_name = "gemini-1.5-flash"
    
    print(f"Testing Gemini with Model: {model_name}")
    print(f"Base URL: {base_url}")
    
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=settings.gemini_api_key.strip()
    )
    
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10,
            timeout=10.0
        )
        print(f"Success! Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
