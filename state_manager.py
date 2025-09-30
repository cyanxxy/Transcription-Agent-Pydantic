"""
Type-safe State Management for ExactTranscriber v2.0
Using Pydantic models for validation
"""

import streamlit as st
from typing import Optional, Any, Dict
import logging

from models import AppState, ProcessingStatus, TranscriptResult, ErrorDetail

logger = logging.getLogger(__name__)


class StateManager:
    """Manages application state with type safety"""

    @staticmethod
    def get_state() -> AppState:
        """Get or initialize application state"""
        if "app_state" not in st.session_state:
            st.session_state.app_state = AppState()
            logger.info("Initialized new app state")
        return st.session_state.app_state

    @staticmethod
    def update_state(**kwargs) -> None:
        """Update state with validation"""
        state = StateManager.get_state()

        for key, value in kwargs.items():
            if hasattr(state, key):
                try:
                    setattr(state, key, value)
                    logger.debug(f"Updated state: {key} = {value}")
                except Exception as e:
                    logger.error(f"Failed to update state {key}: {e}")
            else:
                logger.warning(f"Unknown state key: {key}")

        # Save back to session state
        st.session_state.app_state = state

    @staticmethod
    def reset_state() -> None:
        """Reset state to initial values"""
        state = StateManager.get_state()
        state.reset()
        st.session_state.app_state = state
        logger.info("State reset to initial values")

    @staticmethod
    def set_processing(filename: str) -> None:
        """Set state to processing mode"""
        StateManager.update_state(
            status=ProcessingStatus.PROCESSING,
            current_file=filename,
            processing_progress=0.0,
            error=None,
        )

    @staticmethod
    def set_complete(result: TranscriptResult) -> None:
        """Set state to complete with result"""
        StateManager.update_state(
            status=ProcessingStatus.COMPLETE,
            transcript_result=result,
            processing_progress=1.0,
            error=None,
        )

    @staticmethod
    def set_error(error_message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Set state to error"""
        error = ErrorDetail(
            code="TRANSCRIPTION_ERROR",
            message=error_message,
            category="processing",
            details=details,
            recoverable=True,
        )

        StateManager.update_state(
            status=ProcessingStatus.ERROR, error=error, processing_progress=0.0
        )

    @staticmethod
    def update_progress(progress: float) -> None:
        """Update processing progress"""
        StateManager.update_state(processing_progress=progress)

    @staticmethod
    def is_processing() -> bool:
        """Check if currently processing"""
        state = StateManager.get_state()
        return state.status == ProcessingStatus.PROCESSING

    @staticmethod
    def has_result() -> bool:
        """Check if transcript result is available"""
        state = StateManager.get_state()
        return state.transcript_result is not None

    @staticmethod
    def get_api_key() -> Optional[str]:
        """Get API key from various sources"""
        # Check session state first
        if "api_key" in st.session_state and st.session_state.api_key:
            return st.session_state.api_key

        # Check secrets
        if hasattr(st, "secrets"):
            if "GOOGLE_API_KEY" in st.secrets:
                return st.secrets["GOOGLE_API_KEY"]
            if "GEMINI_API_KEY" in st.secrets:
                return st.secrets["GEMINI_API_KEY"]

        # Check environment
        import os

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            return api_key

        return None

    @staticmethod
    def set_api_key(api_key: str) -> None:
        """Store API key in session state"""
        st.session_state["api_key"] = api_key
        StateManager.update_state(api_key_configured=True)
        logger.info(
            f"API key set: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}"
        )

    @staticmethod
    def clear_api_key() -> None:
        """Clear API key from session state"""
        if "api_key" in st.session_state:
            del st.session_state["api_key"]
        StateManager.update_state(api_key_configured=False)
        logger.info("API key cleared")

    @staticmethod
    def get_model_name() -> str:
        """Get selected model name"""
        state = StateManager.get_state()
        return state.model_name

    @staticmethod
    def set_model_name(model_name: str) -> None:
        """Set model name"""
        StateManager.update_state(model_name=model_name)


# Convenience functions for backward compatibility


def get_state() -> AppState:
    """Get current application state"""
    return StateManager.get_state()


def update_state(**kwargs) -> None:
    """Update application state"""
    StateManager.update_state(**kwargs)


def reset_state() -> None:
    """Reset application state"""
    StateManager.reset_state()
