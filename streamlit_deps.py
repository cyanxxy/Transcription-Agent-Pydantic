"""Streamlit-backed dependency loading helpers.

This keeps Streamlit-specific state access out of the core dependency module.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

from dependencies import AppDeps
from models import AppState

DEFAULT_PARAKEET_MODEL = AppState().parakeet_model


def _session_value(st_module: Any, name: str, default: Any) -> Any:
    session_state = getattr(st_module, "session_state", None)
    app_state = getattr(session_state, "app_state", None) if session_state else None

    if session_state is not None and hasattr(session_state, name):
        return getattr(session_state, name)
    if app_state is not None and hasattr(app_state, name):
        return getattr(app_state, name)
    return default


def build_app_deps_from_streamlit(st_module: Any | None = None) -> Optional[AppDeps]:
    """Build ``AppDeps`` from Streamlit session state, secrets, or env vars."""
    if st_module is None:
        try:
            st_module = importlib.import_module("streamlit")
        except ImportError:
            return None

    api_key = None
    session_state = getattr(st_module, "session_state", None)
    if session_state is not None and hasattr(session_state, "api_key"):
        api_key = session_state.api_key

    if not api_key and hasattr(st_module, "secrets"):
        secrets = st_module.secrets
        api_key = secrets.get("GOOGLE_API_KEY") or secrets.get("GEMINI_API_KEY")

    if not api_key:
        import os

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return None

    model_name = _session_value(st_module, "model_name", "gemini-3-flash-preview")
    judge_model_name = _session_value(
        st_module, "judge_model_name", "gemini-3.1-pro-preview"
    )
    candidate_strategy = _session_value(st_module, "candidate_strategy", "dual_gemini")
    parakeet_model = (
        _session_value(st_module, "parakeet_model", DEFAULT_PARAKEET_MODEL)
        or DEFAULT_PARAKEET_MODEL
    ).strip() or DEFAULT_PARAKEET_MODEL
    auto_format = _session_value(st_module, "auto_format", True)
    remove_fillers = _session_value(st_module, "remove_fillers", False)
    use_judge_pipeline = _session_value(st_module, "use_judge_pipeline", True)
    transcription_thinking_level = _session_value(
        st_module,
        "transcription_thinking_level",
        _session_value(st_module, "thinking_level", "high"),
    )
    judge_thinking_level = _session_value(st_module, "judge_thinking_level", "high")

    return AppDeps.from_config(
        api_key=api_key,
        model_name=model_name,
        judge_model_name=judge_model_name,
        candidate_strategy=candidate_strategy,
        parakeet_model=parakeet_model,
        auto_format=auto_format,
        remove_fillers=remove_fillers,
        use_judge_pipeline=use_judge_pipeline,
        transcription_thinking_level=transcription_thinking_level,
        judge_thinking_level=judge_thinking_level,
    )
