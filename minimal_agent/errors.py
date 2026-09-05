from __future__ import annotations


class AgentError(Exception):
    code = "agent_error"
    recoverable = False

    def __init__(self, message: str, *, recoverable: bool | None = None):
        super().__init__(message)
        if recoverable is not None:
            self.recoverable = recoverable


class LLMError(AgentError):
    code = "llm_error"
    recoverable = True


class ProtocolAgentError(AgentError):
    code = "protocol_error"
    recoverable = True


class ToolNotFoundError(AgentError):
    code = "tool_not_found"
    recoverable = False


class ToolValidationError(AgentError):
    code = "tool_validation_error"
    recoverable = True


class ToolExecutionError(AgentError):
    code = "tool_execution_error"
    recoverable = True


def error_info(exc: Exception) -> dict[str, object]:
    return {
        "error_type": getattr(exc, "code", exc.__class__.__name__),
        "error": str(exc),
        "recoverable": bool(getattr(exc, "recoverable", False)),
    }

