from __future__ import annotations

from types import SimpleNamespace

import main


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class _FakeSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _FakeStreamlit(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()
        self.session_state = _FakeSessionState(
            candidate_strategy="dual_gemini",
            auto_format=True,
            remove_fillers=False,
            use_judge_pipeline=False,
        )
        self.sidebar = _NullContext()

    def markdown(self, *args, **kwargs):
        del args, kwargs

    def radio(self, label, options, index=0, **kwargs):
        del label, kwargs
        return options[index]

    def columns(self, n):
        if isinstance(n, int):
            count = n
        else:
            count = len(n)
        return [_NullContext() for _ in range(count)]

    def checkbox(self, label, value=False, **kwargs):
        del kwargs
        if label == "Use Judge Pipeline":
            return False
        return value

    def caption(self, *args, **kwargs):
        del args, kwargs

    def expander(self, *args, **kwargs):
        del args, kwargs
        return _NullContext()

    def selectbox(self, label, options, index=0, **kwargs):
        del kwargs
        if label == "Format Type":
            return "Auto"
        return options[index]

    def text_input(self, *args, **kwargs):
        del args, kwargs
        return ""

    def text_area(self, *args, **kwargs):
        del args, kwargs
        return ""


def test_render_sidebar_preserves_candidate_strategy_when_judge_pipeline_is_off(
    monkeypatch,
) -> None:
    fake_st = _FakeStreamlit()
    candidate_strategy_calls: list[str] = []

    monkeypatch.setattr(main, "st", fake_st)
    monkeypatch.setattr(main.StateManager, "get_model_name", lambda: "gemini-3-flash-preview")
    monkeypatch.setattr(main.StateManager, "set_model_name", lambda value: None)
    monkeypatch.setattr(main.StateManager, "get_auto_format", lambda: True)
    monkeypatch.setattr(main.StateManager, "get_remove_fillers", lambda: False)
    monkeypatch.setattr(main.StateManager, "set_auto_format", lambda value: None)
    monkeypatch.setattr(main.StateManager, "set_remove_fillers", lambda value: None)
    monkeypatch.setattr(main.StateManager, "get_use_judge_pipeline", lambda: False)
    monkeypatch.setattr(main.StateManager, "set_use_judge_pipeline", lambda value: None)
    monkeypatch.setattr(
        main.StateManager,
        "set_candidate_strategy",
        lambda value: candidate_strategy_calls.append(value),
    )
    monkeypatch.setattr(main.StateManager, "get_candidate_strategy", lambda: "dual_gemini")
    monkeypatch.setattr(
        main.StateManager,
        "get_parakeet_model",
        lambda: "nvidia/parakeet-ctc-0.6b",
    )
    monkeypatch.setattr(main.StateManager, "set_parakeet_model", lambda value: None)

    main.render_sidebar()

    assert fake_st.session_state.candidate_strategy == "dual_gemini"
    assert "single_gemini" not in candidate_strategy_calls


def test_render_sidebar_allows_resetting_parakeet_model_to_default(
    monkeypatch,
) -> None:
    fake_st = _FakeStreamlit()
    set_parakeet_calls: list[str] = []

    monkeypatch.setattr(main, "st", fake_st)
    monkeypatch.setattr(
        main.StateManager, "get_model_name", lambda: "gemini-3-flash-preview"
    )
    monkeypatch.setattr(main.StateManager, "set_model_name", lambda value: None)
    monkeypatch.setattr(main.StateManager, "get_auto_format", lambda: True)
    monkeypatch.setattr(main.StateManager, "get_remove_fillers", lambda: False)
    monkeypatch.setattr(main.StateManager, "set_auto_format", lambda value: None)
    monkeypatch.setattr(main.StateManager, "set_remove_fillers", lambda value: None)
    monkeypatch.setattr(main.StateManager, "get_use_judge_pipeline", lambda: False)
    monkeypatch.setattr(main.StateManager, "set_use_judge_pipeline", lambda value: None)
    monkeypatch.setattr(main.StateManager, "set_candidate_strategy", lambda value: None)
    monkeypatch.setattr(
        main.StateManager, "get_candidate_strategy", lambda: "dual_gemini"
    )
    monkeypatch.setattr(
        main.StateManager,
        "get_parakeet_model",
        lambda: "custom/parakeet-model",
    )
    monkeypatch.setattr(
        main.StateManager,
        "set_parakeet_model",
        lambda value: set_parakeet_calls.append(value),
    )

    main.render_sidebar()

    assert fake_st.session_state.parakeet_model == main.AppState().parakeet_model
    assert set_parakeet_calls == [main.AppState().parakeet_model]
