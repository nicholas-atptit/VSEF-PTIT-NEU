import os
from openai import AsyncOpenAI
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test():
    api_key = os.getenv("GEMINI_API_KEY")
    model = "gemini-1.5-pro"
    
    # Test with models/ prefix
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai"
    )
    
    print(f"Testing model: {model}")
    try:
        # Try without models/ prefix first (auto-added by my logic, but let's test)
        res1 = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
        )
        print(f"Success without models/ prefix: {res1.choices[0].message.content[:20]}")
    except Exception as e:
        print(f"Failed without models/ prefix: {e}")

    try:
        # Try with models/ prefix
        m_with_prefix = f"models/{model}" if not model.startswith("models/") else model
        res2 = await client.chat.completions.create(
            model=m_with_prefix,
            messages=[{"role": "user", "content": "hi"}],
        )
        print(f"Success with models/ prefix: {res2.choices[0].message.content[:20]}")
    except Exception as e:
        print(f"Failed with models/ prefix: {e}")

if __name__ == "__main__":
    asyncio.run(test())
