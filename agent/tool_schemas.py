"""OpenAI-style function schemas for the tools in ``tools/log_tools.py``.

Groq's chat-completions API is OpenAI-compatible, so these are plain
OpenAI "tools" function schemas: each one describes a callable the model may
choose to invoke, along with the JSON shape of its arguments.

``ticket_id`` is deliberately ABSENT from both schemas below. The tool
functions in ``tools/log_tools.py`` require it, but the executor injects it
from ``TaskState.ticket_id`` at call time rather than letting the model
supply it. This means the model can never choose or hallucinate which
incident is being queried — it can only ever act on the ticket the run was
actually started for. That is a deliberate prompt-injection defense: per
docs/design.md, permission tiers are enforced outside the LLM, not by asking
it nicely. Keep it that way — do not add ``ticket_id`` to either schema.
"""

QUERY_LOGS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_logs",
        "description": (
            "Search the log store for one service and return an aggregated summary — "
            "total matching lines, a count per recurring message pattern, and up to 5 "
            "sample lines. Never returns a raw dump. Use this first to find out WHAT is "
            "failing; the dominant pattern usually names the failure mode (e.g. a "
            "connection-pool timeout, an OOM kill). Widening the level (ERROR -> WARN -> "
            "INFO) only adds lower-severity routine noise; it does not reveal new "
            "incident patterns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": (
                        "Name of the service to search logs for, as a lowercase slug, "
                        "e.g. 'checkout', 'auth-api', 'ingress'. Infer it from the "
                        "ticket text."
                    ),
                },
                "window": {
                    "type": "string",
                    "description": (
                        "How far back to search, as a positive integer plus a unit: "
                        "'m' minutes, 'h' hours, 'd' days. Examples: '30m', '2h', '1d'. "
                        "Any other format is rejected."
                    ),
                },
                "level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
                    "description": (
                        "Minimum severity to include; the result contains this level "
                        "and everything above it. Start with 'ERROR' for an incident; "
                        "only drop to 'WARN' or 'INFO' if ERROR came back empty."
                    ),
                },
            },
            "required": ["service", "window", "level"],
            "additionalProperties": False,
        },
    },
}

QUERY_METRICS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_metrics",
        "description": (
            "Fetch a time series for one named metric on one service, returning 12 "
            "points plus min/max/avg/current and the configured capacity ceiling when "
            "one exists. Use this AFTER query_logs to confirm or refute a specific "
            "hypothesis — pick the metric that would prove the failure mode the logs "
            "pointed at. One call returns one metric; call again to check another."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": (
                        "Name of the service to read the metric for, as a lowercase "
                        "slug, e.g. 'checkout', 'auth-api', 'ingress'. Infer it from "
                        "the ticket text."
                    ),
                },
                "metric": {
                    "type": "string",
                    "enum": [
                        "db_pool_active_connections",
                        "cache_used_memory_percent",
                        "ingress_listen_queue_depth",
                        "readiness_probe_success_rate",
                        "auth_token_verification_failure_rate",
                        "disk_usage_percent",
                        "request_latency_p95",
                        "cpu_percent",
                        "memory_percent",
                        "error_rate",
                        "request_count",
                        "network_throughput_mbps",
                    ],
                    "description": (
                        "Which metric to fetch. Choose the one that would confirm your "
                        "current hypothesis: db_pool_active_connections (connections in "
                        "use, against a max of 100) for pool exhaustion; "
                        "cache_used_memory_percent for cache eviction/OOM; "
                        "ingress_listen_queue_depth (against a max of 512) for "
                        "connection backlog at the edge; readiness_probe_success_rate "
                        "for pods failing health checks after a deploy; "
                        "auth_token_verification_failure_rate for token/key rejection; "
                        "disk_usage_percent for a filling volume; request_latency_p95 "
                        "(ms) for general slowness. cpu_percent, memory_percent, "
                        "error_rate, request_count and network_throughput_mbps are "
                        "broad health indicators and rarely isolate a root cause on "
                        "their own."
                    ),
                },
                "window": {
                    "type": "string",
                    "description": (
                        "How far back to fetch, as a positive integer plus a unit: "
                        "'m' minutes, 'h' hours, 'd' days. Examples: '30m', '2h', '1d'. "
                        "Any other format is rejected. Use a window wide enough to "
                        "include the pre-incident baseline, so a step change or "
                        "upward trend is visible."
                    ),
                },
            },
            "required": ["service", "metric", "window"],
            "additionalProperties": False,
        },
    },
}

TOOL_SCHEMAS = [QUERY_LOGS_SCHEMA, QUERY_METRICS_SCHEMA]

# Args every tool function in tools/log_tools.py requires but which are
# supplied by the executor from TaskState, never by the model — kept out of
# the schemas above so the model has no way to choose or hallucinate them.
EXECUTOR_INJECTED_ARGS = ("ticket_id",)
