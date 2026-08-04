"""Observability setup.

Configures OpenTelemetry to export traces to Azure Monitor / Application
Insights. Both processes (runner and evaluator) call :func:`setup_observability`
so their spans land in the same App Insights resource and can be joined by id.

The ``service_name`` is surfaced as ``cloud_RoleName`` in App Insights, so the
runner-created ``invoke_agent`` span and the evaluator-created
``gen_ai.evaluation.input`` span are attributable to distinct roles while
sharing one trace.
"""

from __future__ import annotations

import os

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.trace import Tracer


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def setup_observability(service_name: str) -> Tracer:
    """Wire up Azure Monitor export and return a tracer for *service_name*.

    Reads ``APPLICATIONINSIGHTS_CONNECTION_STRING`` for the destination and
    ``ENABLE_SENSITIVE_DATA`` to opt in to capturing prompts/responses.

    Args:
        service_name: Logical role for this process (e.g. ``"eval-runner"`` or
            ``"eval-worker"``); surfaces as ``cloud_RoleName``.

    Returns:
        An OpenTelemetry :class:`~opentelemetry.trace.Tracer` for this process.

    Raises:
        RuntimeError: If the Application Insights connection string is not set.
    """
    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        raise RuntimeError(
            "APPLICATIONINSIGHTS_CONNECTION_STRING is not set. "
            "Copy .env.example to .env and fill it in."
        )

    resource = Resource.create({SERVICE_NAME: service_name})
    configure_azure_monitor(
        connection_string=connection_string,
        resource=resource,
    )

    # Agent Framework instrumentation is on by default once OTel providers
    # exist, but sensitive data (prompts/responses) must be explicitly enabled.
    try:
        from agent_framework.observability import enable_sensitive_telemetry

        if _truthy(os.environ.get("ENABLE_SENSITIVE_DATA")):
            enable_sensitive_telemetry()
    except ImportError:
        # agent_framework is only required in the runner process.
        pass

    return trace.get_tracer(service_name)
