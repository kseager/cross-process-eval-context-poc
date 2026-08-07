"""Observability setup.

Configures OpenTelemetry to export traces (and metrics/logs) to Azure Monitor /
Application Insights, and turns on Agent Framework instrumentation so that agent
invocations, chat calls, and tool calls are automatically traced.

Both the evaluation driver and the agent-service call
:func:`setup_observability` so their spans land in the same App Insights
resource and share one trace. Each process sets ``OTEL_SERVICE_NAME`` before
calling it, so ``create_resource`` tags their spans with distinct
``cloud_RoleName`` values, keeping each process attributable to its role.
"""

from __future__ import annotations

import os

from azure.monitor.opentelemetry import configure_azure_monitor


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def setup_observability() -> None:
    """Wire up Azure Monitor exporters and Agent Framework instrumentation.

    Reads ``APPLICATIONINSIGHTS_CONNECTION_STRING`` for the Application Insights
    destination and ``ENABLE_SENSITIVE_DATA`` to opt in to capturing prompts and
    responses in traces.

    Raises:
        RuntimeError: If the Application Insights connection string is not set.
    """
    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        raise RuntimeError(
            "APPLICATIONINSIGHTS_CONNECTION_STRING is not set. "
            "Copy .env.example to .env and fill it in."
        )

    from agent_framework.observability import create_resource, enable_sensitive_telemetry

    configure_azure_monitor(
        connection_string=connection_string,
        resource=create_resource(),
    )

    # Bind an EventLoggerProvider onto Azure Monitor's LoggerProvider; without
    # one, emitted events are silently dropped and never reach App Insights.
    from opentelemetry._events import set_event_logger_provider
    from opentelemetry._logs import get_logger_provider
    from opentelemetry.sdk._events import EventLoggerProvider

    set_event_logger_provider(EventLoggerProvider(get_logger_provider()))

    if _truthy(os.environ.get("ENABLE_SENSITIVE_DATA")):
        enable_sensitive_telemetry()
