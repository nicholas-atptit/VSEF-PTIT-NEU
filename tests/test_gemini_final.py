import asyncio
from openai import AsyncOpenAI
from config.settings import get_settings

async def main():
    settings = get_settings()
    # Updated Gemini model and base URL
    model_name = settings.gemini_model_name
    base_url = settings.gemini_base_url
    
    # Try with and without prefix for robustness
    test_models = [model_name]
    if not model_name.startswith("models/"):
        test_models.append(f"models/{model_name}")

    for model in test_models:
        print(f"\nTesting Gemini with Model: {model}")
        print(f"Base URL: {base_url}")
        
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=settings.gemini_api_key.strip()
        )
        
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say OK if you are gemini."}],
                max_tokens=20,
                timeout=15.0
            )
            print(f"Success! Response: {response.choices[0].message.content}")
            return # Exit if any succeeds
        except Exception as e:
            print(f"Failed for {model}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
