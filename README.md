# ExactTranscriber

Audio transcription application built with Pydantic AI and Google Gemini 2.5 models. Features speaker diarization, automatic chunking for large files, and comprehensive quality analysis.

## Features

- **Transcription** - Gemini 2.5 Flash/Pro with speaker diarization
- **Smart Chunking** - Automatic splitting of large files with context preservation
- **Context-Aware** - Provide speaker names, technical terms, and domain context
- **Quality Metrics** - Readability, vocabulary richness, speaker consistency scoring
- **Editing Tools** - Auto-formatting, find/replace, capitalization fixes
- **Multiple Exports** - TXT, SRT (subtitles), JSON

## Requirements

- Python 3.11+
- FFmpeg (for audio processing)
- [Google Gemini API key](https://aistudio.google.com/app/apikey)

## Installation

```bash
git clone https://github.com/cyanxxy/Transcription-Agent-Pydantic.git
cd Transcription-Agent-Pydantic

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

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
3. Click "Start Transcription"
4. View results with timestamps and speakers

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
- **SRT** - Standard subtitle format for video players
- **JSON** - Structured data with metadata and quality metrics

## Project Structure

```
├── agents/
│   ├── transcription_agent.py   # Pydantic AI agent (core transcription)
│   ├── context_agent.py         # Context processing utilities
│   ├── quality_validator.py     # Quality metrics calculation
│   └── editing_tools.py         # Text editing operations
├── main.py                      # Streamlit web interface
├── workflow.py                  # Pipeline orchestration
├── models.py                    # Pydantic data models
├── dependencies.py              # Dependency injection configs
├── state_manager.py             # Session state management
└── requirements.txt
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                          │
│                      (main.py)                           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               TranscriptionWorkflow                      │
│                   (workflow.py)                          │
│  Pipeline orchestration, state management, error handling│
└───────┬─────────────┬─────────────┬─────────────────────┘
        │             │             │
        ▼             ▼             ▼
┌───────────────┬───────────┬───────────────┐
│ Transcription │  Quality  │    Editing    │
│    Agent      │  Metrics  │    Tools      │
├───────────────┼───────────┼───────────────┤
│ Pydantic AI   │ Scoring   │ Format/Replace│
│ Gemini API    │ Validation│ Capitalization│
└───────────────┴───────────┴───────────────┘
        │
        ▼
┌───────────────┐
│  Gemini API   │
│  Flash / Pro  │
└───────────────┘
```

### Design Principles

- **Single Agent Pattern** - One Pydantic AI agent for transcription, utility functions for other operations
- **Direct Dependency Injection** - Dependencies passed directly to functions via `deps` parameter
- **Multimodal Input** - Audio sent via `BinaryContent` through `Agent.run()`
- **No Tool Decorators** - Utilities called directly by workflow, not as agent tools

## Configuration

### Model Selection

| Model | Speed | Cost | Use Case |
|-------|-------|------|----------|
| `gemini-2.5-flash` | Fast | ~$0.005/min | General use |
| `gemini-2.5-pro` | Slower | ~$0.08/min | High accuracy |

### Processing Settings

Edit `dependencies.py`:

```python
@dataclass
class TranscriptionDeps:
    model_name: str = "gemini-2.5-flash"
    max_file_size_mb: int = 200
    chunk_duration_ms: int = 120000    # 2 minutes
    chunk_overlap_ms: int = 5000       # 5 seconds overlap
    preserve_context: bool = True
    auto_format: bool = True
    remove_fillers: bool = False       # Remove um, uh, like
```

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
- `streamlit==1.32.0` - Web UI
- `google-genai>=1.10.0` - Gemini SDK

**Audio:**
- `pydub==0.25.1` - Audio processing
- `aiofiles>=23.2.1` - Async file I/O

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Configuration error | Verify `GOOGLE_API_KEY` in secrets or environment |
| File size exceeds limit | Compress: `ffmpeg -i input.mp3 -b:a 128k output.mp3` |
| Failed to load audio | Check format, ensure FFmpeg installed (`ffmpeg -version`) |
| Import errors | `pip install --force-reinstall -r requirements.txt` |
| Poor transcription quality | Add context, use Pro model, check audio clarity |

## License

MIT
