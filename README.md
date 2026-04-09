# ExactTranscriber

ExactTranscriber is a Streamlit transcription app built with Pydantic AI and Google Gemini 3. The current architecture is a judge pipeline:

1. gather user context
2. generate one or more transcript candidates
3. run a judge agent to select or merge the best final transcript
4. optionally align timestamps with Parakeet

## Features

- Multi-agent transcript judging with Pydantic AI
- Candidate strategies:
  - `single_gemini`
  - `dual_gemini`
  - `gemini_plus_parakeet`
- Optional Parakeet ASR candidate and post-judge timestamp alignment
- Context-aware prompting for speakers, topic, terms, and format
- Automatic chunking for long audio with chunk-level judging
- Quality scoring, editing tools, and TXT/SRT/JSON export

## Requirements

- Python 3.11+
- FFmpeg
- Google Gemini API key
- `pydantic-ai>=1.67.0,<2`
- `google-genai>=1.67.0,<2`
- Optional: `nemo_toolkit[asr]`, `torch`, and `torchaudio` for Parakeet
- Python 3.13+ installs also pull in `audioop-lts` automatically for `pydub`

## Installation

```bash
git clone https://github.com/cyanxxy/Transcription-Agent-Pydantic.git
cd Transcription-Agent-Pydantic

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

This project now expects a fresh Python 3.11+ environment. Older Python 3.9
virtualenvs will not be able to install the current `pydantic-ai` and
`google-genai` releases.

For Parakeet support:

```bash
pip install -r requirements-full.txt
```

## Configuration

Set a Gemini API key with either Streamlit secrets or environment variables. The app accepts either `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

```toml
# .streamlit/secrets.toml
GOOGLE_API_KEY = "your-key-here"
GEMINI_API_KEY = "your-key-here"
```

Or:

```bash
export GOOGLE_API_KEY="your-key-here"
export GEMINI_API_KEY="your-key-here"
```

## Run

```bash
streamlit run main.py
```

## How It Works

### Judge Pipeline

The default pipeline is:

```text
audio + context
  -> candidate transcribers
  -> judge agent
  -> optional Parakeet timestamp review
  -> final transcript
```

When audio is chunked, the app runs candidate generation and judging per chunk, then merges the judged chunks back into a single transcript.

### Candidate Strategies

| Strategy | What runs |
| --- | --- |
| `single_gemini` | One Gemini transcription candidate, then judge cleanup |
| `dual_gemini` | Primary Gemini model plus the paired secondary Gemini model, then judge selection/merge |
| `gemini_plus_parakeet` | Gemini plus Parakeet ASR as transcript candidates, then judge selection/merge |

### Models

Supported Gemini models:

- `gemini-3-flash-preview`
- `gemini-3.1-flash-lite-preview`
- `gemini-3.1-pro-preview`

The primary model is chosen in the UI. The judge agent uses `gemini-3.1-pro-preview`
by default, and the UI label is `3.1 Pro (Judge/Quality)`.
For backward compatibility, the app normalizes the deprecated
`gemini-3-pro-preview` name to `gemini-3.1-pro-preview`.
`dual_gemini` uses the following mapping:

- `gemini-3-flash-preview` -> `gemini-3.1-flash-lite-preview`
- `gemini-3.1-flash-lite-preview` -> `gemini-3-flash-preview`
- `gemini-3.1-pro-preview` -> `gemini-3-flash-preview`

Advanced config can set `transcription_thinking_level` and `judge_thinking_level`
independently through `AppDeps.from_config()`. The legacy `thinking_level`
alias is still accepted for transcription compatibility.

## UI Usage

1. Upload an audio file.
2. Choose the primary Gemini model.
3. Enable `Use Judge Pipeline`.
4. Choose a candidate strategy. The sidebar labels the options `Single Gemini + Judge`, `Dual Gemini + Judge`, and `Gemini + Parakeet + Judge`.
5. Optionally add topic, speaker names, and technical terms.
6. Start transcription.

The result view shows:

- final transcript
- judge notes
- candidate summaries
- quality metrics
- export options

## Project Structure

```text
.
├── agents/
│   ├── judge_agent.py
│   ├── transcription_agent.py
│   ├── timestamp_tool.py
│   ├── context_agent.py
│   ├── quality_validator.py
│   └── editing_tools.py
├── dependencies.py
├── main.py
├── models.py
├── state_manager.py
├── workflow.py
└── tests/
```

## Important Files

- `workflow.py`: candidate fan-out, judge fan-in, timestamp review
- `agents/judge_agent.py`: judge agent definition
- `agents/transcription_agent.py`: Gemini transcription agent and chunking
- `agents/timestamp_tool.py`: Parakeet candidate transcription and alignment
- `dependencies.py`: runtime config and strategy resolution
- `models.py`: result, candidate, and judge data models

## Development

Lint:

```bash
ruff check . --exclude tests --exclude __pycache__
```

Format:

```bash
black --exclude '/(\.git|\.venv|__pycache__|\.streamlit|tests)/' .
```

Type check:

```bash
mypy . --ignore-missing-imports
```

Tests:

```bash
python3 -m pytest tests/
```

## Notes

- Parakeet is optional. If it is not installed, `gemini_plus_parakeet` will degrade gracefully and the app will keep Gemini candidates.
- A legacy direct single-model mode still exists when `Use Judge Pipeline` is disabled.
- `AppDeps.from_config()` still accepts the legacy `use_orchestrator` flag and maps it to `use_judge_pipeline` for backward compatibility.
