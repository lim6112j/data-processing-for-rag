"""The one connection path both OpenRouter roles share.

T2 (the adjudicator) and the schema proposer are different roles with the
same dependency posture: both talk to OpenRouter through the OpenAI SDK,
both must resolve the key the SDK does not read (``OPENROUTER_API_KEY``),
and both must fail with an actionable message when the ``openai`` package
is missing -- not a bare ImportError from inside the SDK.

Having this in one place is what stops the roles from drifting: a third
LLM touchpoint inherits the same key resolution and the same failure
wording, and no role can quietly develop its own credential handling.
"""

from __future__ import annotations

import os
from typing import Any

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def resolve_client(
    client: Any | None,
    *,
    api_key: str | None,
    base_url: str = OPENROUTER_BASE_URL,
    timeout: float | None = None,
    feature: str = "this feature",
) -> Any:
    """Return the injected client, or build one from the environment.

    The injected client always wins -- tests pass fakes, and a configured
    client means the operator already decided how to reach the network.
    Otherwise the SDK is imported lazily so the core pipeline keeps working
    without it, the key comes from ``OPENROUTER_API_KEY`` unless given
    explicitly, and a missing package or missing key is reported together:
    the operator should fix one trip, not discover the second problem on
    the next run.
    """
    if client is not None:
        return client

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    try:
        from openai import OpenAI
    except ImportError as exc:
        parts = [f"{feature} needs the 'openai' package: uv add openai"]
        if key is None:
            parts.append(
                "and an API key: set OPENROUTER_API_KEY in the environment"
            )
        raise RuntimeError("; ".join(parts)) from exc

    kwargs: dict[str, Any] = {"base_url": base_url, "api_key": key}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)


__all__ = ["OPENROUTER_BASE_URL", "resolve_client"]
