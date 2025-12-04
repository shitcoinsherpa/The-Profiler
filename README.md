# Behavioral Profiling System

Multimodal AI-powered psychological assessment tool that analyzes video content to generate behavioral profiles.

Created by [@LLMSherpa](https://x.com/LLMSherpa)

## What It Does

Upload a video (or paste a YouTube/Twitter/TikTok URL) and the system runs it through a 6-stage analysis pipeline:

1. **Video** - Loads and validates the video file
2. **Audio** - Extracts audio track and transcribes speech
3. **Visual** - Analyzes facial expressions, body language, behavioral archetypes
4. **Multimodal** - Cross-references visual and audio cues, detects performance vs authenticity
5. **Voice** - Examines vocal patterns, accent, paralinguistic markers
6. **Synthesis** - Combines all analyses into a final behavioral profile

## Screenshots

### Main Interface
![Main Interface](https://github.com/shitcoinsherpa/The-Profiler/blob/main/screenshots/main1.png)

Upload a video file or import from URL. The interface shows video metadata (duration, resolution, file size) and validates requirements before processing. A 6-step progress bar tracks the analysis pipeline in real-time.

### Analysis Components - Visual & Multimodal
![Visual & Multimodal Analysis](https://github.com/shitcoinsherpa/The-Profiler/blob/main/screenshots/analysis1.png)

The **Behavioral Archetype** section identifies the subject's primary archetype (e.g., "The Narcissistic Operator") with timestamped behavioral evidence. The **Multimodal Analysis** classifies the interaction type (staged performance vs genuine), analyzes camera awareness, and performs cross-modal synchronization between voice tone and facial expressions.

### Analysis Components - Audio & Linguistic
![Audio & Linguistic Analysis](https://github.com/shitcoinsherpa/The-Profiler/blob/main/screenshots/analysis2.png)

**Voice Forensics** profiles the speaker's accent, speech patterns, and cultural influences (e.g., "Internet/Content Creator Culture" cadence). **LIWC Linguistic Analysis** examines word choice, pronoun usage, and cognitive complexity markers.

### FBI Profile & Visualizations
![Profile with Charts](https://github.com/shitcoinsherpa/The-Profiler/blob/main/screenshots/fbi1.png)

The synthesis tab presents interactive visualizations:
- **Confidence Gauge** - Overall analysis confidence score
- **Big Five Radar** - Personality dimensions (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
- **Dark Triad Bars** - Narcissism, Machiavellianism, Psychopathy scores (0-100)
- **Threat Matrix** - Risk levels for manipulation, volatility, compliance, etc.

Below the charts, the **FBI Behavioral Synthesis** provides an executive summary, operational recommendations, and interview approach strategies.

## Requirements

- Python 3.10+
- FFmpeg (must be in PATH or project directory)
- OpenRouter API key ([get one here](https://openrouter.ai/keys))

## Installation

```bash
git clone https://github.com/shitcoinsherpa/The-Profiler.git
cd The-Profiler
```

**Windows:**
```batch
setup.bat   # Creates venv, installs dependencies
run.bat     # Starts the app
```

**Linux/macOS:**
```bash
chmod +x setup.sh run.sh
./setup.sh  # Creates venv, installs dependencies
./run.sh    # Starts the app
```

On first launch, enter your OpenRouter API key in the Settings accordion.

## Usage

1. Upload a video (10-300 seconds, max 100MB) or paste a URL
2. Optionally enter a subject name to track profiles over time
3. Click **Initiate Behavioral Analysis**
4. Watch the 6-step progress bar as analysis runs (~1-2 minutes)
5. Review results in Analysis Components and FBI Profile tabs
6. Export as PDF or JSON

## Features

- **Native video processing** via Gemini's multimodal understanding
- **Parallel analysis pipeline** for faster results
- **Result caching** to avoid redundant API calls
- **Profile history** with search and comparison
- **Custom prompt templates** for fine-tuning analysis
- **PDF report generation**

## License

MIT

## Disclaimer

For educational and research purposes only.
