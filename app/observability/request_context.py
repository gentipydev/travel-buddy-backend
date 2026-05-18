"""Request-scoped ContextVars.

`request_id_var` is set by `request_logging_middleware` for every HTTP
request; `user_id_var` is populated by the Supabase auth dependency
(`app/auth/supabase_auth.py`) once a JWT is verified.

Both vars are read by `RequestContextFilter` (in logging_config) so every
log line carries them without each call site passing `extra=...`.
"""
from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
