"""Agent construction (unchanged from a normal Agent Framework app).

This is deliberately identical to an ordinary Foundry-backed agent: the whole
point of the POC is that the *agent* process needs **no** special code for
span-exact evaluation correlation. The runner owns the ``evaluation_context``
span and
injects trace context; the agent just runs.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential

DEFAULT_INSTRUCTIONS = (
    "You are a helpful assistant. Answer the user's question as accurately and "
    "concisely as possible."
)


class _CompatFoundryChatClient(FoundryChatClient):
    """FoundryChatClient that does not opt into encrypted reasoning content.

    For stateless (single-turn) requests the base OpenAI Responses client
    auto-appends ``include=["reasoning.encrypted_content"]`` to preserve
    reasoning across turns. Some Foundry project/model combinations reject that
    with ``400 "Encrypted content is not supported with this model."``. The
    Foundry *agent* client strips this automatically; the *chat* client does
    not, so we mirror that strip here. This is a service-compatibility shim, not
    evaluation logic.
    """

    async def _prepare_options(
        self,
        messages: Sequence[Any],
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        run_options = await super()._prepare_options(messages, options)
        caller_requested = "reasoning.encrypted_content" in (options.get("include") or [])
        include = run_options.get("include")
        if not caller_requested and isinstance(include, list):
            stripped = [i for i in include if i != "reasoning.encrypted_content"]
            if stripped:
                run_options["include"] = stripped
            else:
                run_options.pop("include", None)
        return run_options


def build_agent(
    name: str = "GroundTruthAgent",
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> Agent:
    """Create a Foundry-backed agent.

    Reads ``FOUNDRY_PROJECT_ENDPOINT`` and ``AZURE_AI_MODEL_DEPLOYMENT_NAME``
    (or ``FOUNDRY_MODEL_NAME``) from the environment. Auth uses
    ``DefaultAzureCredential`` (run ``az login``).

    Returns:
        A ready-to-run :class:`~agent_framework.Agent`.

    Raises:
        RuntimeError: If required environment variables are missing.
    """
    project_endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
    if not project_endpoint:
        raise RuntimeError(
            "FOUNDRY_PROJECT_ENDPOINT is not set. "
            "Copy .env.example to .env and fill it in."
        )

    model_name = os.environ.get(
        "AZURE_AI_MODEL_DEPLOYMENT_NAME"
    ) or os.environ.get("FOUNDRY_MODEL_NAME")
    if not model_name:
        raise RuntimeError(
            "Model deployment name is not set. Set "
            "AZURE_AI_MODEL_DEPLOYMENT_NAME or FOUNDRY_MODEL_NAME."
        )

    client = _CompatFoundryChatClient(
        project_endpoint=project_endpoint,
        model=model_name,
        credential=DefaultAzureCredential(),
    )

    return Agent(
        client=client,
        name=name,
        instructions=instructions,
    )
