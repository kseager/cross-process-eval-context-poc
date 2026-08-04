"""Agent construction (unchanged from a normal Agent Framework app).

This is deliberately identical to an ordinary Foundry-backed agent: the whole
point of the POC is that the *agent* process needs **no** special code for
span-exact evaluation correlation. The runner owns the ``invoke_agent`` span and
injects trace context; the agent just runs.
"""

from __future__ import annotations

import os

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential

DEFAULT_INSTRUCTIONS = (
    "You are a helpful assistant. Answer the user's question as accurately and "
    "concisely as possible."
)


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

    client = FoundryChatClient(
        project_endpoint=project_endpoint,
        model=model_name,
        credential=DefaultAzureCredential(),
    )

    return Agent(client=client, name=name, instructions=instructions)
