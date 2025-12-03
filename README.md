# FBI-Style Behavioral Profiling System

Multimodal AI-Powered Psychological Assessment Tool

Created by [@LLMSherpa](https://x.com/LLMSherpa)

## Overview

This system performs comprehensive behavioral analysis on video content using Gemini's native video understanding through OpenRouter. It analyzes visual, audio, and linguistic cues to generate detailed psychological profiles.

![Main Interface](https://github.com/shitcoinsherpa/The-Profiler/blob/main/main1.png)

## Features

- **Native Video Processing**: Gemini 3 Pro analyzes video directly (no frame extraction)
- **Modular Analysis Pipeline**: Focused sub-analyses run in parallel for reliability
- **Visual Analysis**: FACS coding, behavioral archetypes, body language, deception indicators
- **Audio Analysis**: Voice characteristics, sociolinguistic profiling, vocal deception markers
- **Multimodal Integration**: Cross-modal timestamp synchronization, gesture-verbal alignment
- **Comprehensive Synthesis**: Big Five personality, Dark Triad assessment, threat matrix, clinical differentials
- **Interactive Visualizations**: Plotly-powered charts for personality profiles and threat assessment

## Screenshots

### Analysis Components
Individual analysis stages including Visual Essence, Multimodal Behavioral, Audio/Voice, and LIWC Linguistic analysis:

![Analysis Components - Visual & Multimodal](https://github.com/shitcoinsherpa/The-Profiler/blob/main/analysis1.png)

![Analysis Components - Audio & LIWC](https://github.com/shitcoinsherpa/The-Profiler/blob/main/analysis2.png)

### FBI Profile & Insights
Synthesized behavioral profile with visual analytics including Big Five personality radar, Dark Triad assessment, and Threat Matrix:

![FBI Profile & Visualizations](https://github.com/shitcoinsherpa/The-Profiler/blob/main/fbi1.png)

## Requirements

- Python 3.10+
- FFmpeg (must be in PATH or in project directory)
- OpenRouter API key (get one at https://openrouter.ai/keys)

## Installation

1. Clone the repository
2. Run `setup.bat` to create virtual environment and install dependencies
3. Run `run.bat` to start the Gradio interface
4. Enter your OpenRouter API key in the Settings tab (securely stored locally)

## Usage

1. Upload a video file (10-300 seconds, max 100MB)
2. Select AI models for each analysis stage (defaults to Gemini 3 Pro Preview)
3. Click "Initiate Behavioral Analysis"
4. View results across multiple tabs

## API Server

For programmatic access, run the REST API server:

```bash
python api_server.py --host 0.0.0.0 --port 8000
```

API documentation available at `http://localhost:8000/docs`

## Architecture

The system uses a modular analysis pipeline with native video processing:

```
Video Input
    |
    v
Native Video Loading (Base64)
    |
    v
+-------------------+-------------------+-------------------+
|   Visual Stage    |  Multimodal Stage |   Audio Stage     |
| - FACS Coding     | - Gesture Timeline| - Voice Profile   |
| - Archetype ID    | - Cross-Modal Sync| - Sociolinguistic |
| - Body Language   | - Environment     | - Deception Voice |
| - Deception       | - Camera Aware    |                   |
+-------------------+-------------------+-------------------+
    |                       |                   |
    +-----------------------+-------------------+
                            |
                            v
                    Synthesis Stage
                - Personality (Big Five + Dark Triad)
                - Threat Assessment Matrix
                - Clinical Differential Diagnosis
                - Contradiction Resolution
                - Red Team Self-Critique
                            |
                            v
                    Final Integration
```

## License

MIT License

## Disclaimer

This tool is for educational and research purposes. Any use for actual behavioral profiling should comply with applicable laws and ethical guidelines.
