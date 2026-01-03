# ExactTranscriber v2.0

Advanced audio transcription application powered by Pydantic AI orchestration and Google Gemini 2.5 models.

## Features

- 🎙️ **High-Quality Transcription** - Gemini 2.5 Flash/Pro models with speaker diarization
- 🧩 **Smart Chunking** - Automatic processing of large files with context preservation
- 🎯 **Context-Aware** - User-provided context for speaker names, technical terms, and domain knowledge
- 📊 **Quality Metrics** - Comprehensive quality scoring and analysis
- ✏️ **Editing Tools** - Auto-formatting, find/replace, capitalization fixes
- 💾 **Multiple Exports** - TXT, SRT (subtitles), and JSON formats
- 🚀 **Real-time Processing** - Async pipeline with progress tracking

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       Streamlit UI Layer                         │
│                          (main.py)                               │
│  • File Upload  • Progress Tracking  • Results Display          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TranscriptionWorkflow                         │
│                        (workflow.py)                             │
│  • Pipeline orchestration with dependency injection              │
│  • Direct function calls to utilities                            │
│  • State management and error handling                           │
└─────┬───────────┬──────────────┬────────────┬───────────────────┘
      │           │              │            │
      ▼           ▼              ▼            ▼
┌──────────┬──────────────┬──────────────┬──────────────┐
│Transcribe│   Context    │   Quality    │   Editing    │
│  Module  │   Module     │   Module     │   Module     │
├──────────┼──────────────┼──────────────┼──────────────┤
│•Validate │•Speakers     │•Metrics      │•Format       │
│•Chunk    │•Terms        │•Scoring      │•Replace      │
│•Process  │•Prompts      │•Validation   │•Capitalize   │
│•Merge    │•Domain       │•Assessment   │•Cleanup      │
└──────────┴──────────────┴──────────────┴──────────────┘
      │
      └───────────────────┬───────────────────────
                          │
                          ▼
                  ┌────────────────┐
                  │  Gemini API    │
                  │  Flash / Pro   │
                  │ (Direct SDK)   │
                  └────────────────┘
```

### Technical Design

**Core Pattern**: Single Pydantic AI Agent for orchestration + Utility functions for operations

**TranscriptionAgent** (Pydantic AI Agent)
- High-level orchestration with structured output
- Type-safe dependency injection via `deps` parameter
- Direct Gemini API integration via `google-genai` SDK (new GA SDK)
- Audio file upload and transcription generation

**Utility Modules** (Pure Python Functions)
- Context processing (speaker mapping, prompt enhancement)
- Quality metrics calculation (readability, consistency, scoring)
- Editing operations (formatting, find/replace, capitalization)

**Dependency Injection**
- `TranscriptionDeps`, `QualityDeps`, `EditingDeps`, `ExportDeps`
- Passed directly to utility functions for type safety
- Centralized configuration management

### Data Flow

```
Audio File
   │
   ├──► validate_audio_file(deps, file_data, filename)
   │    └──► Returns: validation result + temp path
   │
   ├──► process_audio_file(deps, temp_path)
   │    └──► Returns: AudioMetadata (duration, format, chunking needs)
   │
   ├──► [If user context provided]
   │    create_context_prompt(speakers, topic, terms, ...)
   │    └──► Returns: Enhanced prompt string
   │
   ├──► [If needs_chunking]
   │    │
   │    ├──► chunk_audio(deps, audio_path)
   │    │    └──► Returns: List of chunk files with timing info
   │    │
   │    ├──► For each chunk:
   │    │    run_transcription_agent(agent, deps, chunk_path, prompt, chunk_info, previous_context, speakers)
   │    │    └──► Agent.run([... BinaryContent(audio) ...]) → Gemini response → Typed segments
   │    │
   │    └──► merge_chunks(deps, all_segments)
   │         └──► Returns: Merged TranscriptSegments with dedupe
   │
   ├──► [If no chunking]
   │    run_transcription_agent(agent, deps, audio_path, prompt, None, None, speakers)
   │    └──► Agent.run([... BinaryContent(audio) ...]) → Gemini response → Typed segments
   │
   ├──► [If speaker_names provided]
   │    map_speakers_to_context(segments, speaker_names)
   │    └──► Returns: Segments with mapped speaker names
   │
   ├──► calculate_quality_metrics(deps, segments)
   │    └──► Returns: Quality scores and warnings
   │
   ├──► calculate_overall_score(deps, metrics)
   │    └──► Returns: Overall quality score (0-100)
   │
   └──► TranscriptResult
        └──► segments + metadata + quality + processing_time
             └──► Export: TXT / SRT / JSON
```

### Key Implementation Details

1. **Single Pydantic AI Agent**: Only `TranscriptionAgent` is a Pydantic AI Agent, created with `Agent(model_name, deps_type, output_type, system_prompt)`

2. **Multimodal Agent Invocation**: Audio bytes are wrapped in `BinaryContent` and sent through `Agent.run`, so Pydantic AI handles the Gemini 2.5 request end-to-end:
   ```python
   prompt = build_transcription_prompt(...)
   result = await agent.run(
       [prompt, BinaryContent(data=audio_bytes, media_type="audio/wav")],
       deps=ctx.deps,
       model_settings=_build_google_settings(ctx.deps),
   )
   segments = result.output  # -> List[TranscriptSegment]
   ```

3. **Direct Dependency Injection**: All utility functions receive `deps` directly for type-safe dependency injection:
   ```python
   async def validate_audio_file(
       deps: TranscriptionDeps,
       file_data: bytes,
       filename: str
   ) -> Dict[str, Any]:
       # Access dependencies directly
       if size_mb > deps.max_file_size_mb:
           ...
   ```

4. **No Tool Decorators**: No `@agent.tool` or similar - utilities are called directly by workflow

5. **Model Name Format**: Uses string format `google-gla:gemini-2.5-flash` instead of model objects

## Installation

### Prerequisites

- Python 3.11+
- FFmpeg (for audio processing)
- Google Gemini API key ([Get one free](https://aistudio.google.com/app/apikey))

### Setup

```bash
# Clone repository
git clone https://github.com/cyanxxy/Transcription-Agent-Pydantic.git
cd Transcription-Agent-Pydantic

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key (Option 1: Streamlit secrets)
mkdir -p .streamlit
cat > .streamlit/secrets.toml << EOF
GOOGLE_API_KEY = "your-gemini-api-key-here"
EOF

# Or Option 2: Environment variable
export GOOGLE_API_KEY="your-gemini-api-key-here"

# Run application
streamlit run main.py
```

The app will open at `http://localhost:8501`

## Project Structure

```
├── agents/                      # Processing modules
│   ├── __init__.py             # Module exports
│   ├── transcription_agent.py  # Core transcription (Pydantic AI Agent)
│   ├── context_agent.py        # Context processing utilities
│   ├── quality_validator.py    # Quality metrics utilities
│   └── editing_tools.py        # Editing operation utilities
├── main.py                     # Streamlit web interface
├── workflow.py                 # Pipeline orchestration
├── models.py                   # Pydantic data models
├── dependencies.py             # Dependency injection configs
├── state_manager.py            # Session state management
├── styles.py                   # UI styling
├── utils.py                    # Helper utilities
└── requirements.txt            # Python dependencies
```

## Usage

### Basic Transcription

1. Open the app: `streamlit run main.py`
2. Upload an audio file (MP3, WAV, M4A, FLAC, OGG)
3. Click "Start Transcription"
4. View results with timestamps and speakers

### With Context (Recommended)

Expand the "Add context for better accuracy" section in the sidebar:

- **Topic/Domain**: e.g., "Medical consultation", "Tech interview", "Legal deposition"
- **Speaker Names**: Comma-separated names (e.g., "John, Sarah, Dr. Smith")
- **Technical Terms**: Line-separated terms to watch for
- **Format Type**: Meeting, Interview, Lecture, Podcast, Legal, Medical

This significantly improves accuracy for domain-specific content and speaker identification.

### Editing Tools

After transcription, use the **Edit** tab:

- **Auto-Format**: Fix spacing, punctuation, and apply sentence case
- **Fix Capitalization**: Capitalize proper nouns and sentence starts
- **Find & Replace**: Bulk text replacements with case-sensitive/whole-word options

### Export Options

**Export** tab provides multiple formats:

- **Text (.txt)**: Plain text with timestamps and speakers
- **Subtitles (.srt)**: Standard subtitle format for video players
- **JSON (.json)**: Complete structured data with metadata and quality metrics

## Configuration

### Model Selection

- **Flash (Fast)**: `gemini-2.5-flash` - Quick processing, lower cost
- **Pro (Quality)**: `gemini-2.5-pro` - Higher accuracy, more expensive

Toggle in the sidebar under Settings.

### Processing Options

- **Auto-format**: Apply formatting rules automatically
- **No fillers**: Remove filler words (um, uh, like, etc.)

### Advanced Configuration

Edit `dependencies.py` to customize:

```python
@dataclass
class TranscriptionDeps:
    api_key: str
    model_name: str = "gemini-2.5-flash"
    max_file_size_mb: int = 200
    chunk_duration_ms: int = 120000  # 2 minutes
    chunk_overlap_ms: int = 5000     # 5 seconds
    preserve_context: bool = True
    auto_format: bool = True
    # ... more settings
```

## Supported Audio Formats

| Format | Extension | Max Size | Notes |
|--------|-----------|----------|-------|
| MP3 | .mp3 | 200MB | Most common |
| WAV | .wav | 200MB | Uncompressed |
| M4A | .m4a | 200MB | Apple format |
| FLAC | .flac | 200MB | Lossless |
| OGG | .ogg | 200MB | Open format |

**Chunking**: Files longer than 2 minutes are automatically split with 5-second overlap for context preservation.

## Quality Metrics

The app calculates comprehensive quality scores:

- **Overall Score** (0-100): Weighted combination of all metrics
- **Readability** (0-100): Based on Flesch reading ease
- **Vocabulary Richness** (0-100): Unique word ratio
- **Sentence Variety** (0-100): Length variation
- **Punctuation Density** (0-1): Punctuation usage
- **Timestamp Coverage** (0-100%): Segment timestamp presence
- **Speaker Consistency** (0-100%): Label consistency

Quality assessments:
- 80-100: Excellent
- 60-79: Good
- 40-59: Fair
- 0-39: Poor

## Development

### Code Quality

```bash
# Linting with Ruff
ruff check . --exclude tests --exclude __pycache__

# Auto-fix issues
ruff check . --fix --exclude tests --exclude __pycache__

# Format with Black
black --exclude '/(\.git|\.venv|__pycache__|\.streamlit|tests)/' .

# Type checking with MyPy
mypy . --ignore-missing-imports
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run tests
python -m pytest tests/
```

### Project Dependencies

**Core**:
- `pydantic-ai>=0.0.50` - Agent framework with Google Gemini support
- `streamlit==1.32.0` - Web UI
- `pydantic>=2.5.0` - Data validation

**Audio Processing**:
- `pydub==0.25.1` - Audio manipulation
- `aiofiles>=23.2.1` - Async file I/O
- `nest-asyncio>=1.5.6` - Nested event loop support

**Development**:
- `ruff>=0.1.0` - Fast Python linter
- `black>=23.0.0` - Code formatter
- `pytest>=8.2.2` - Testing framework

## API Costs

Google Gemini pricing (as of 2024):

**Gemini 2.5 Flash**:
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens
- Audio: ~$0.005 per minute

**Gemini 2.5 Pro**:
- Input: $1.25 per 1M tokens
- Output: $5.00 per 1M tokens
- Audio: ~$0.08 per minute

The app shows estimated costs before processing.

## Limitations

- Maximum file size: 200MB (Gemini API limit)
- Audio must be clear enough for Gemini to process
- Speaker diarization quality depends on voice distinctiveness
- No real-time/streaming transcription (batch processing only)
- Context window limits may affect very long files (handled via chunking)

## Troubleshooting

### "Configuration error. Please check setup."

- Ensure `GOOGLE_API_KEY` is set in `.streamlit/secrets.toml` or environment
- Verify API key is valid at [Google AI Studio](https://aistudio.google.com/app/apikey)

### "File size exceeds limit"

- Reduce audio quality/bitrate to compress file under 200MB
- Use FFmpeg: `ffmpeg -i input.mp3 -b:a 128k output.mp3`

### "Failed to load audio file"

- Verify file format is supported (MP3, WAV, M4A, FLAC, OGG)
- Check file isn't corrupted by playing in audio player
- Ensure FFmpeg is installed: `ffmpeg -version`

### Transcription quality issues

- Add context via sidebar (speakers, topic, technical terms)
- Use Gemini 2.5 Pro instead of Flash for better accuracy
- Ensure audio quality is good (minimal background noise)
- Check speaker separation in multi-speaker audio

### Import errors

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Verify installations
python -c "import pydantic_ai; from google import genai; print('OK')"
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run linting: `ruff check . && black .`
5. Commit: `git commit -am 'Add feature'`
6. Push: `git push origin feature-name`
7. Create Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with [Pydantic AI](https://ai.pydantic.dev/) framework
- Powered by [Google Gemini 2.5](https://deepmind.google/technologies/gemini/)
- UI with [Streamlit](https://streamlit.io/)

---

**Version**: 2.0
**Author**: Original repo by [cyanxxy](https://github.com/cyanxxy/Transcription-Agent-Pydantic)
**Status**: Production-ready with full Pydantic AI integration
