# ExactTranscriber

Advanced audio transcription application using Pydantic AI agent orchestration and Google Gemini models.

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit UI Layer                       │
│                         (main.py)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TranscriptionWorkflow                         │
│                    (workflow.py)                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Orchestrates agent communication and data flow          │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────┬───────────┬──────────────┬───────────┬────────────────┘
         │           │              │           │
         ▼           ▼              ▼           ▼
┌──────────────┬──────────────┬──────────────┬──────────────┐
│Transcription │   Context    │   Quality    │   Editing    │
│    Agent     │    Agent     │   Agent      │    Agent     │
├──────────────┼──────────────┼──────────────┼──────────────┤
│▪ File Valid. │▪ Speaker ID  │▪ Scoring     │▪ Formatting  │
│▪ Chunking    │▪ Terms       │▪ Readability │▪ Find/Replace│
│▪ Gemini API  │▪ Domain      │▪ Consistency │▪ Segments    │
│▪ Merging     │▪ Prompts     │▪ Validation  │▪ Cleanup     │
└──────────────┴──────────────┴──────────────┴──────────────┘
         │           │              │           │
         └───────────┴──────────────┴───────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  Gemini 2.5    │
                    │  Flash/Pro     │
                    └────────────────┘
```

### Data Flow

```
Audio Input ──► TranscriptionAgent
                      │
                      ├──► Validation
                      ├──► Chunking (if > 2min)
                      └──► Processing
                            │
                            ▼
              ContextAgent Enhancement
                      │
                      ├──► User Context
                      ├──► Domain Terms
                      └──► Speaker Names
                            │
                            ▼
                   Gemini 2.5 API
                      │
                      ├──► Transcription
                      └──► Thinking Budget
                            │
                            ▼
                   QualityAgent
                      │
                      ├──► Quality Score
                      ├──► Metrics
                      └──► Validation
                            │
                            ▼
                   EditingAgent
                      │
                      ├──► Auto-format
                      ├──► Corrections
                      └──► Final Output
                            │
                            ▼
                Export (TXT/SRT/JSON)
```

### Agent Communication

```
Dependencies (Pydantic Models)
        │
        ├──► TranscriptionDeps
        ├──► ContextDeps
        ├──► QualityDeps
        └──► EditingDeps
                │
                ▼
    Agent Tools (Decorated Functions)
        │
        ├──► @agent.tool decorators
        ├──► Type-safe I/O models
        └──► Agent execution framework
                │
                ▼
    RunContext[DepsType]
        │
        ├──► Dependency Injection
        └──► GoogleModel integration
```

### Agent Details

**TranscriptionAgent**
- Audio file validation and metadata extraction
- Intelligent chunking for large files (up to 200MB)
- Context preservation between chunks
- Gemini model integration with thinking capabilities

**QualityAgent**
- Transcript quality scoring (0-100 scale)
- Readability and coherence analysis
- Speaker consistency validation
- Grammar and formatting detection

**ContextAgent**
- User context processing (speakers, topics, terminology)
- Dynamic prompt enhancement
- Domain-specific vocabulary handling

**EditingAgent**
- Auto-formatting and punctuation correction
- Filler word removal
- Find/replace operations
- Segment manipulation

### Technical Implementation

**Core Technologies**
- Pydantic AI for agent orchestration
- Google Gemini Flash/Pro models
- Pydantic V2 for type safety
- AsyncIO for concurrent operations
- Streamlit for web interface

**Key Design Patterns**
- Agent-based architecture with Pydantic AI
- Dependency injection for configuration management
- Tool-based agent composition
- Session state persistence
- Retry logic with exponential backoff

## Installation

```bash
# Clone repository
git clone https://github.com/cyanxxy/Transcription-Agent-Pydantic.git
cd Transcription-Agent-Pydantic

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your Gemini API key

# Run application
streamlit run main.py
```

## Project Structure

```
├── agents/                 # Pydantic AI agents
│   ├── transcription_agent.py
│   ├── quality_validator.py
│   ├── context_agent.py
│   └── editing_tools.py
├── main.py                # Streamlit application
├── workflow.py            # Workflow orchestration
├── models.py              # Data models
├── dependencies.py        # Dependency injection
├── state_manager.py       # State management
└── utils.py              # Utilities
```

## Features

**Processing Capabilities**
- Chunked processing for large files
- Context-aware transcription
- Quality assessment with metrics
- Multiple export formats (TXT, SRT, JSON)

**Model Features**
- Configurable temperature and token limits
- Structured JSON output support
- Automatic retry on failures
- Real-time cost estimation
- Support for both Flash and Pro models

**User Context Support**
- Speaker identification
- Technical terminology
- Domain-specific formatting
- Custom instructions

## Configuration

Create `.streamlit/secrets.toml`:
```toml
GOOGLE_API_KEY = "your-gemini-api-key"
```

Or use environment variable:
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

## Requirements

- Python 3.11+
- FFmpeg for audio processing (automatically installed on most systems)
- Google Gemini API key (get one free at https://aistudio.google.com/app/apikey)

## Getting Started

1. Get your free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Follow the installation steps above
3. Run `streamlit run main.py`
4. Upload an audio file and start transcribing!

## Supported Audio Formats

- MP3, WAV, M4A, FLAC, OGG
- Automatic format detection and conversion
- Maximum file size: 200MB
- Automatic chunking for files longer than 2 minutes
- Cross-chunk context preservation for continuity

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Quality
```bash
# Linting
ruff check .

# Type checking
mypy . --ignore-missing-imports
```

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT