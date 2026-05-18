"""Anthropic refine-tool registry.

`REFINE_TOOLS` is the list passed to `client.beta.messages.stream(...,
tools=...)` on refine turns. Order is only for prompt clarity; Anthropic
doesn't care about the position of a tool within the list.

`REFINE_TOOL_NAMES` is the set the chat router checks when filtering
tool_use blocks out of the assistant's final message.

`apply_refine_tool` is the validate-and-apply dispatcher; see
`dispatcher.py`.
"""
from app.tools.refine.add_day import ADD_DAY_TOOL
from app.tools.refine.delete_day import DELETE_DAY_TOOL
from app.tools.refine.delete_stop import DELETE_STOP_TOOL
from app.tools.refine.dispatcher import apply_refine_tool
from app.tools.refine.insert_stop import INSERT_STOP_TOOL
from app.tools.refine.reorder_stops import REORDER_STOPS_TOOL
from app.tools.refine.update_stop import UPDATE_STOP_TOOL
from app.tools.refine.update_trip import UPDATE_TRIP_TOOL


# Order matters only for prompt clarity — Anthropic doesn't care.
REFINE_TOOLS = [
    UPDATE_STOP_TOOL,
    INSERT_STOP_TOOL,
    DELETE_STOP_TOOL,
    REORDER_STOPS_TOOL,
    ADD_DAY_TOOL,
    DELETE_DAY_TOOL,
    UPDATE_TRIP_TOOL,
]


REFINE_TOOL_NAMES = {tool["name"] for tool in REFINE_TOOLS}


__all__ = [
    "ADD_DAY_TOOL",
    "DELETE_DAY_TOOL",
    "DELETE_STOP_TOOL",
    "INSERT_STOP_TOOL",
    "REFINE_TOOLS",
    "REFINE_TOOL_NAMES",
    "REORDER_STOPS_TOOL",
    "UPDATE_STOP_TOOL",
    "UPDATE_TRIP_TOOL",
    "apply_refine_tool",
]
