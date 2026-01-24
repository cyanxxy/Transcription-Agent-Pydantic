# ExactTranscriber

Audio transcription application built with Pydantic AI and Google Gemini 3 models. Features a **truly agentic orchestrator** that autonomously decides what tools to use, NVIDIA Parakeet timestamp correction, speaker diarization, automatic chunking, and comprehensive quality analysis.

## Features

- **Autonomous AI Agent** - Agent analyzes results and decides whether to fix timestamps (not user-controlled)
- **Smart Decision Making** - Agent explains reasoning: "Skipped timestamp correction because score=92"
- **Accurate Timestamps** - NVIDIA Parakeet + NeMo Forced Aligner for word-level timing (when needed)
- **Quality Transcription** - Gemini 3 Flash/Pro with speaker diarization
- **Smart Chunking** - Automatic splitting of large files with context preservation
- **Context-Aware** - Provide speaker names, technical terms, and domain context
- **Quality Metrics** - Readability, vocabulary richness, speaker consistency scoring
- **Editing Tools** - Auto-formatting, find/replace, capitalization fixes
- **Multiple Exports** - TXT, SRT (subtitles), JSON

## Requirements

- Python 3.11+
- FFmpeg (for audio processing)
- [Google Gemini API key](https://aistudio.google.com/app/apikey)
- CUDA-capable GPU (optional, for faster Parakeet processing)

## Installation

```bash
git clone https://github.com/cyanxxy/Transcription-Agent-Pydantic.git
cd Transcription-Agent-Pydantic

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Basic installation (Gemini transcription only)
pip install -r requirements.txt

# OR full installation with Parakeet timestamp correction
pip install -r requirements-full.txt
```

### Optional: NVIDIA Parakeet for Accurate Timestamps

If you installed with `requirements.txt` (basic), add Parakeet later:

```bash
pip install nemo_toolkit[asr] torch torchaudio
```

**Note:** NeMo requires ~3GB disk space for models. A CUDA GPU is recommended but not required. The agent will gracefully skip timestamp correction if NeMo is not installed.

### API Key Configuration

**Option 1** - Streamlit secrets:
```bash
mkdir -p .streamlit
echo 'GOOGLE_API_KEY = "your-key-here"' > .streamlit/secrets.toml
```

**Option 2** - Environment variable:
```bash
export GOOGLE_API_KEY="your-key-here"
```

### Run

```bash
streamlit run main.py
```
Opens at `http://localhost:8501`

## Usage

### Basic Transcription

1. Upload audio file (MP3, WAV, M4A, FLAC, OGG)
2. Select model (Flash for speed, Pro for accuracy)
3. Enable "Agentic Mode" (recommended)
4. Click "Start Transcription"
5. View results with timestamps, speakers, and agent's reasoning

The autonomous agent:
- Transcribes audio with Gemini for quality text
- **Analyzes** timestamp quality (gaps, drift, coverage)
- **Decides** whether to fix timestamps based on analysis
- Explains its decision: "Skipped because score=87" or "Fixed because irregular gaps detected"
- Validates final transcript quality

### Adding Context

Expand "Add context for better accuracy" in sidebar:

| Field | Description | Example |
|-------|-------------|---------|
| Topic/Domain | Subject area | Medical consultation, Tech interview |
| Speaker Names | Comma-separated | John, Sarah, Dr. Smith |
| Technical Terms | Line-separated | Domain-specific vocabulary |
| Format Type | Content style | Meeting, Interview, Lecture, Podcast |

Context significantly improves speaker identification and terminology accuracy.

### Editing

After transcription, use the Edit tab:
- **Auto-Format** - Fix spacing, punctuation, sentence case
- **Fix Capitalization** - Proper nouns and sentence starts
- **Find & Replace** - Bulk replacements with case/whole-word options

### Export

Available formats in Export tab:
- **TXT** - Plain text with timestamps and speakers
- **SRT** - Standard subtitle format with speaker labels (for video players)
- **JSON** - Structured data with metadata and quality metrics

## Project Structure

```
├── agents/
│   ├── orchestrator_agent.py   # Autonomous orchestrator (4 tools)
│   │   ├── transcribe_audio()      # Gemini transcription
│   │   ├── analyze_timestamps()    # Quality analysis for decisions
│   │   ├── fix_timestamps()        # Parakeet correction (agent decides)
│   │   └── check_quality()         # Final validation
│   ├── transcription_agent.py  # Gemini transcription agent
│   ├── timestamp_tool.py       # Parakeet NFA implementation
│   ├── context_agent.py        # Context processing utilities
│   ├── quality_validator.py    # Quality metrics calculation
│   └── editing_tools.py        # Text editing operations
├── main.py                     # Streamlit web interface
├── workflow.py                 # Pipeline orchestration
├── models.py                   # Pydantic data models
├── dependencies.py             # Dependency injection configs
├── state_manager.py            # Session state management
└── requirements.txt
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (main.py)                    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│               TranscriptionWorkflow (workflow.py)            │
│          Pipeline orchestration, state management            │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              ORCHESTRATOR AGENT (Autonomous)                 │
│   "I analyze results and DECIDE what tools to use"           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                    @agent.tool (4 tools)
        ┌──────────┬──────────┼──────────┬──────────┐
        ▼          ▼          ▼          ▼          │
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│transcribe_ │ │ analyze_   │ │ fix_       │ │ check_     │
│audio()     │ │ timestamps │ │ timestamps │ │ quality()  │
│            │ │ ()         │ │ ()         │ │            │
│ Always     │ │ Agent      │ │ Agent      │ │ Always     │
│ runs       │ │ analyzes   │ │ DECIDES    │ │ runs       │
│            │ │ quality    │ │ to call    │ │            │
└────────────┘ └────────────┘ └────────────┘ └────────────┘
                    │               ▲
                    │   Decision    │
                    └───────────────┘
                    score < 70 → FIX
                    score > 85 → SKIP
```

### Autonomous Decision Flow

```
1. transcribe_audio()     → Get Gemini transcription
2. analyze_timestamps()   → Score: 0-100, recommendation: fix/skip/optional
3. IF score < 70:         → fix_timestamps() + note "Fixing because..."
   ELSE:                  → skip + note "Skipping because score=X"
4. check_quality()        → Final validation
5. Return with reasoning in processing_notes
```

### Design Principles

- **Truly Agentic** - Agent analyzes and decides, not user checkboxes
- **Explainable Decisions** - Agent provides reasoning in `processing_notes`
- **Tool-Based Orchestration** - Orchestrator uses `@agent.tool` decorators
- **Direct Dependency Injection** - Dependencies passed directly via `deps` parameter
- **Graceful Degradation** - Works without NeMo, agent decides to skip timestamp fixing

## Configuration

### Model Selection

| Model | Speed | Cost | Use Case |
|-------|-------|------|----------|
| `gemini-3-flash-preview` | Fast | $0.50/$3.00 per 1M tokens | General use |
| `gemini-3-pro-preview` | Slower | $2.00/$12.00 per 1M tokens (<200k tokens) | High accuracy |

### Processing Settings

Edit `dependencies.py`:

```python
@dataclass
class TranscriptionDeps:
    model_name: str = "gemini-3-flash-preview"
    thinking_level: str = "high"       # minimal, low, medium (Flash), high
    max_file_size_mb: int = 200
    chunk_duration_ms: int = 120000    # 2 minutes
    chunk_overlap_ms: int = 5000       # 5 seconds overlap
    preserve_context: bool = True
    auto_format: bool = True
    remove_fillers: bool = False       # Remove um, uh, like

    # Orchestrator settings
    use_orchestrator: bool = True      # Enable agentic orchestration
    parakeet_model: str = "nvidia/parakeet-ctc-0.6b"
```

### Agentic Mode Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `use_orchestrator` | `True` | Enable autonomous agent (recommended) |
| `parakeet_model` | `nvidia/parakeet-ctc-0.6b` | NeMo ASR model for alignment |

**Note:** In agentic mode, the agent autonomously decides whether to use Parakeet based on its analysis of timestamp quality. There's no manual toggle - the agent explains its decision in `processing_notes`.

## Supported Formats

| Format | Extension | Max Size | Notes |
|--------|-----------|----------|-------|
| MP3 | .mp3 | 200MB | Most common |
| WAV | .wav | 200MB | Uncompressed |
| M4A | .m4a | 200MB | Apple format |
| FLAC | .flac | 200MB | Lossless |
| OGG | .ogg | 200MB | Open format |

Files longer than 2 minutes are automatically chunked with 5-second overlap for context preservation.

## Quality Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| Overall Score | 0-100 | Weighted combination of all metrics |
| Readability | 0-100 | Based on Flesch reading ease |
| Vocabulary Richness | 0-100 | Unique word ratio |
| Sentence Variety | 0-100 | Length variation |
| Timestamp Coverage | 0-100% | Segments with valid timestamps |
| Speaker Consistency | 0-100% | Label stability across transcript |

**Interpretation:** 80-100 Excellent | 60-79 Good | 40-59 Fair | <40 Poor

## Development

### Code Quality

```bash
# Lint
ruff check . --exclude tests --exclude __pycache__

# Auto-fix
ruff check . --fix --exclude tests --exclude __pycache__

# Format
black --exclude '/(\.git|\.venv|__pycache__|\.streamlit|tests)/' .

# Type check
mypy . --ignore-missing-imports
```

### Testing

```bash
python -m pytest tests/ -v
```

### Dependencies

**Core:**
- `pydantic-ai>=0.0.50` - Agent framework
- `pydantic>=2.5.0` - Data validation
- `streamlit>=1.32.0` - Web UI
- `google-genai>=1.10.0` - Gemini SDK

**Audio:**
- `pydub==0.25.1` - Audio processing
- `aiofiles>=23.2.1` - Async file I/O

**Timestamp Correction (Optional):**
- `nemo_toolkit[asr]>=2.0.0` - NVIDIA NeMo for Forced Alignment
- `torch>=2.0.0` - PyTorch backend
- `torchaudio>=2.0.0` - Audio I/O

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Configuration error | Verify `GOOGLE_API_KEY` in secrets or environment |
| File size exceeds limit | Compress: `ffmpeg -i input.mp3 -b:a 128k output.mp3` |
| Failed to load audio | Check format, ensure FFmpeg installed (`ffmpeg -version`) |
| Import errors | `pip install --force-reinstall -r requirements.txt` |
| Poor transcription quality | Add context, use Pro model, check audio clarity |
| NeMo not found | Install with `pip install nemo_toolkit[asr] torch torchaudio` |
| Slow timestamp correction | Use CUDA GPU (agent will still decide when to use Parakeet) |
| Agent always skips timestamps | This is normal if Gemini timestamps are good (score > 85) |

**Multi-user deployments:** API keys are stored in process-wide environment variables. For hosted multi-tenant deployments, implement per-request API key handling.

## How It Works

### Autonomous Orchestrator Flow

1. **User uploads audio** - File validated and processed
2. **Agent receives task** - "Transcribe this audio file"
3. **transcribe_audio()** - Gemini produces quality text with context awareness
4. **analyze_timestamps()** - Agent examines timestamp quality:
   - Checks for irregular gaps (>20% flagged)
   - Detects timestamp drift
   - Calculates alignment score (0-100)
5. **Agent DECIDES** based on analysis:
   - Score < 70 → "Decided to fix timestamps because score is low"
   - Score > 85 → "Skipped timestamp correction, timestamps look good"
   - Audio < 30s → "Skipped, audio too short for overhead"
6. **fix_timestamps()** - Only called if agent decides it's needed
7. **check_quality()** - Validates final metrics
8. **Result returned** - Transcript + agent's reasoning in `processing_notes`

### Agent Decision Criteria

| Condition | Decision | Reasoning |
|-----------|----------|-----------|
| Alignment score < 70 | FIX | "Timestamps need correction" |
| Alignment score > 85 | SKIP | "Timestamps look good" |
| Irregular gaps > 20% | FIX | "Irregular gaps detected" |
| Timestamp drift detected | FIX | "Timestamps exceed audio duration" |
| Audio duration < 30s | SKIP | "Audio too short, overhead not worth it" |

### Why Two Models?

| Model | Strength | Weakness |
|-------|----------|----------|
| **Gemini** | Context understanding, speaker identification, terminology | Approximate timestamps |
| **Parakeet** | Word-level timestamp accuracy | No context understanding |

The agent intelligently combines both: uses Gemini for quality text, then **decides** if Parakeet is needed for timestamps.

## License

MIT
