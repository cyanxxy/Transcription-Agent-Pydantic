# ExactTranscriber

Audio transcription application built with Pydantic AI agents and Google Gemini 2.5.

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
        ├──► @agent.tool
        └──► Type-safe I/O
                │
                ▼
    RunContext[DepsType]
        │
        └──► Dependency Injection
```

### Agent Details

**TranscriptionAgent**
- Audio file validation and metadata extraction
- Intelligent chunking for large files (up to 200MB)
- Context preservation between chunks
- Direct integration with Gemini 2.5 API

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
- Google Gemini 2.5 Flash/Pro models
- Pydantic V2 for type safety
- AsyncIO for non-blocking operations
- Streamlit for web interface

**Key Design Patterns**
- Dependency injection for configuration
- Tool-based agent architecture
- State management with session persistence
- Structured error handling with retry logic

## Installation

```bash
# Clone repository
git clone https://github.com/[yourusername]/ExactTranscriber.git
cd ExactTranscriber

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

**Gemini 2.5 Integration**
- Thinking budget optimization
- 1M token input support
- Structured output generation
- Cost estimation

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
- FFmpeg for audio processing
- Google Gemini API key

## License

MIT