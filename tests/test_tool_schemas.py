"""Drift guard: keeps agent/tool_schemas.py in sync with tools/log_tools.py and
tools/fake_data.py so the model's view of the tools never silently diverges
from what the executor actually calls.
"""

import inspect

from agent.tool_schemas import (
    EXECUTOR_INJECTED_ARGS,
    QUERY_LOGS_SCHEMA,
    QUERY_METRICS_SCHEMA,
    TOOL_SCHEMAS,
)
from tools.fake_data import METRIC_PROFILES
from tools.log_tools import SEVERITY_ORDER, query_logs, query_metrics


def _param_names(func) -> set[str]:
    return set(inspect.signature(func).parameters)


def _properties(schema) -> dict:
    return schema["function"]["parameters"]["properties"]


def test_query_logs_schema_matches_function_params():
    props = _properties(QUERY_LOGS_SCHEMA)
    assert set(props) | set(EXECUTOR_INJECTED_ARGS) == _param_names(query_logs)


def test_query_metrics_schema_matches_function_params():
    props = _properties(QUERY_METRICS_SCHEMA)
    assert set(props) | set(EXECUTOR_INJECTED_ARGS) == _param_names(query_metrics)


def test_query_logs_all_exposed_params_required():
    props = _properties(QUERY_LOGS_SCHEMA)
    required = set(QUERY_LOGS_SCHEMA["function"]["parameters"]["required"])
    assert required == set(props)


def test_query_metrics_all_exposed_params_required():
    props = _properties(QUERY_METRICS_SCHEMA)
    required = set(QUERY_METRICS_SCHEMA["function"]["parameters"]["required"])
    assert required == set(props)


def test_ticket_id_absent_from_both_schemas():
    assert "ticket_id" not in _properties(QUERY_LOGS_SCHEMA)
    assert "ticket_id" not in _properties(QUERY_METRICS_SCHEMA)


def test_metric_enum_matches_metric_profiles_keys():
    enum = QUERY_METRICS_SCHEMA["function"]["parameters"]["properties"]["metric"]["enum"]
    assert set(enum) == set(METRIC_PROFILES)
    assert len(enum) == len(set(enum))


def test_level_enum_matches_severity_order_exactly():
    enum = QUERY_LOGS_SCHEMA["function"]["parameters"]["properties"]["level"]["enum"]
    assert enum == SEVERITY_ORDER


def test_tool_schemas_list_names_and_type():
    names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    assert names == [
        "query_logs",
        "query_metrics",
        "search_runbooks",
        "search_past_incidents",
        "update_ticket",
    ]
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
