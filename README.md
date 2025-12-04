# Behavioral Profiling System

Multimodal AI-powered psychological assessment tool that analyzes video content to generate behavioral profiles.

Created by [@LLMSherpa](https://x.com/LLMSherpa)

![Main Interface](https://github.com/shitcoinsherpa/The-Profiler/blob/main/screenshots/main1.png)

## What It Does

Upload a video (or paste a YouTube/Twitter/TikTok URL) and the system analyzes:
- **Visual cues**: Facial expressions, body language, behavioral archetypes
- **Audio cues**: Voice characteristics, speech patterns, paralinguistic markers
- **Linguistic cues**: Word choice, pronoun usage, cognitive complexity

Results are synthesized into a comprehensive behavioral profile with personality assessments (Big Five, Dark Triad) and threat evaluation.

## Screenshots

### Analysis Components
![Visual & Multimodal Analysis](https://github.com/shitcoinsherpa/The-Profiler/blob/main/screenshots/analysis1.png)

![Audio & Linguistic Analysis](https://github.com/shitcoinsherpa/The-Profiler/blob/main/screenshots/analysis2.png)

### FBI Profile & Visualizations
![Profile with Charts](https://github.com/shitcoinsherpa/The-Profiler/blob/main/screenshots/fbi1.png)

## Requirements

- Python 3.10+
- FFmpeg (must be in PATH or project directory)
- OpenRouter API key ([get one here](https://openrouter.ai/keys))

## Installation

```bash
# Clone the repo
git clone https://github.com/shitcoinsherpa/The-Profiler.git
cd The-Profiler

# Run setup (creates venv, installs dependencies)
setup.bat

# Start the app
run.bat
```

On first launch, enter your OpenRouter API key in Settings.

## Usage

1. Upload a video file (10-300 seconds, max 100MB) or paste a URL
2. Optionally adjust model selection in Settings
3. Click **Initiate Behavioral Analysis**
4. View results in the Analysis Components and FBI Profile tabs
5. Export as PDF or JSON

## Features

- **Native video processing** via Gemini's video understanding
- **Parallel analysis pipeline** for faster results
- **Result caching** to avoid redundant API calls
- **Profile history** with SQLite storage
- **Custom prompt templates** for fine-tuning analysis
- **Interactive visualizations** (Plotly charts)
- **PDF report generation**

## Project Structure

```
app.py              # Gradio UI (main entry point)
profiler.py         # Analysis pipeline orchestration
modular_executor.py # Parallel sub-analysis execution
modular_prompts.py  # Focused analysis prompts
api_client.py       # OpenRouter API wrapper
visualizations.py   # Plotly chart generation
```

## License

MIT

## Disclaimer

For educational and research purposes only. Any use for actual behavioral profiling should comply with applicable laws and ethical guidelines.
