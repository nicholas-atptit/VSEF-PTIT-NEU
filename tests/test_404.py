import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from openai import AsyncOpenAI


def _mock_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


async def _exercise_gemini_openai_probe() -> dict[str, str]:
    api_key = os.getenv("GEMINI_API_KEY")
    model = "gemini-1.5-pro"
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )

    outputs: dict[str, str] = {}
    for candidate in (model, f"models/{model}"):
        response = await client.chat.completions.create(
            model=candidate,
            messages=[{"role": "user", "content": "hi"}],
        )
        outputs[candidate] = response.choices[0].message.content

    return outputs


def test_gemini_openai_compatible_prefix_probe_offline(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    mock_client = MagicMock(spec=AsyncOpenAI)
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[_mock_response("ok"), _mock_response("ok")]
    )

    with patch.object(sys.modules[__name__], "AsyncOpenAI", return_value=mock_client) as patched_client:
        outputs = asyncio.run(_exercise_gemini_openai_probe())

    patched_client.assert_called_once_with(
        api_key="test-gemini-key",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )
    assert outputs == {
        "gemini-1.5-pro": "ok",
        "models/gemini-1.5-pro": "ok",
    }
    assert mock_client.chat.completions.create.await_count == 2
