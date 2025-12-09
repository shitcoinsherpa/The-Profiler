# The Profiler - FBI-Style Behavioral Analysis System

AI-powered psychological profiling tool that analyzes video content to generate comprehensive behavioral assessments using multimodal analysis.

Created by [@LLMSherpa](https://x.com/LLMSherpa)

## What It Does

Upload a video file (or paste a YouTube/Twitter/TikTok URL) and the system runs a 6-stage analysis pipeline:

| Step | Name | Description |
|------|------|-------------|
| 1 | **Video** | Load, validate, and prepare video for processing |
| 2 | **Audio** | Extract audio track and transcribe speech |
| 3 | **Visual** | Analyze facial expressions, body language, behavioral archetypes |
| 4 | **Multi** | Cross-reference visual and audio cues, detect performance vs authenticity |
| 5 | **Voice** | Examine vocal patterns, accent, paralinguistic markers |
| 6 | **Synth** | Combine all analyses into final FBI-style behavioral profile |

## Demo Video

[https://github.com/user-attachments/assets/c3c3fdb4-9e44-4236-a38d-f485360005bb](https://github.com/shitcoinsherpa/The-Profiler/blob/main/screenshots/analysis2.png)

## Screenshots

### Main Interface
![Main Interface](https://github.com/shitcoinsherpa/The-Profiler/blob/main/screenshots/main1.png)

Upload a video or import from URL. The 6-step progress bar tracks analysis in real-time.

### Analysis Components
![Analysis Components](https://github.com/shitcoinsherpa/The-Profiler/blob/main/screenshots/analysis1.png)

Expandable sections for each analysis type: Behavioral Archetype, Multimodal Analysis, Audio/Voice, LIWC Linguistics, and Speech Transcript.

### FBI Profile & Visualizations
![FBI Profile](https://github.com/shitcoinsherpa/The-Profiler/blob/main/screenshots/fbi1.png)

Interactive charts including:
- **Confidence Gauge** - Overall analysis confidence
- **Big Five Radar** - OCEAN personality dimensions
- **Dark Triad Bars** - Narcissism, Machiavellianism, Psychopathy scores
- **Threat Matrix** - Risk assessment levels
- **MBTI Profile** - Cognitive function preferences

### NCI Deception Analysis
Chase Hughes / NCI University methodology for behavioral deception indicators:
- **BTE Score** - Behavioral Table of Elements deception probability
- **Blink Rate Analysis** - Stress indicator tracking
- **FATE Model** - Focus, Authority, Tribe, Emotion motivational drivers
- **Integrated Deception Assessment** with conflict resolution

## Requirements

- **Python 3.10+** (3.11 recommended)
- **FFmpeg** (required for audio extraction)
- **OpenRouter API key** - [Get one here](https://openrouter.ai/keys)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/shitcoinsherpa/The-Profiler.git
cd The-Profiler
```

### Step 2: Run Setup

The setup script will automatically download FFmpeg and install all dependencies.

**Windows:**
```batch
setup.bat
```

**Linux/macOS:**
```bash
chmod +x setup.sh run.sh
./setup.sh
```

### Step 3: Start the App

**Windows:**
```batch
run.bat
```

**Linux/macOS:**
```bash
./run.sh
```

The app will open in your browser at `http://localhost:7860`

### Step 4: Configure API Key

1. In the web UI, expand **Settings & Configuration**
2. Paste your OpenRouter API key
3. Click **Save API Key**
4. Click **Test Connection** to verify

## Usage

### Basic Workflow

1. **Upload** a video file (10-300 seconds, max 100MB) or paste a URL
2. **Optionally** enter a subject name to track profiles over time
3. Click **Initiate Behavioral Analysis**
4. Watch the 6-step progress bar (~1-3 minutes depending on video length)
5. Review results in the three output tabs:
   - **Analysis Components** - Individual analysis breakdowns
   - **FBI Profile & Insights** - Synthesized profile with visualizations
   - **NCI Deception Analysis** - Chase Hughes methodology indicators
6. **Export** as PDF or JSON

### Output Tabs Explained

| Tab | Contents |
|-----|----------|
| **Analysis Components** | Behavioral Archetype, Multimodal Analysis, Audio/Voice, LIWC Linguistics, Transcript |
| **FBI Profile & Insights** | Personality charts (Big Five, Dark Triad, MBTI, Threat Matrix), FBI Synthesis, Confidence scores |
| **NCI Deception Analysis** | BTE Score gauge, Blink Rate chart, FATE Model radar, Deception Conflict Matrix |

### Profile History

Below the main interface, the **Profile History** section lets you:
- Browse previously analyzed subjects
- Compare multiple profiles for the same person
- Review past analyses without re-running

### Settings

In the **Settings & Configuration** accordion:

| Setting | Description |
|---------|-------------|
| **API Key** | Your OpenRouter API key (encrypted storage) |
| **Vision Model** | Model for visual analysis (default: Gemini 2.5 Pro) |
| **Multimodal Model** | Model for audio+visual analysis (default: Gemini 2.5 Flash) |
| **Synthesis Model** | Model for final profile synthesis (default: Gemini 2.5 Pro) |
| **Max Resolution** | Video processing resolution (720p/1080p) |
| **Frame Interval** | Seconds between frame samples |
| **Cache** | View/clear cached results |

### Custom Prompt Templates

In the **Custom Prompt Templates** accordion, you can:
- Create custom analysis prompts
- Save templates for different use cases
- Modify the default FBI synthesis prompt

## Features

- **Native Video Processing** - Sends video directly to Gemini's multimodal understanding
- **Parallel Analysis Pipeline** - Multiple sub-analyses run concurrently for speed
- **Result Caching** - Avoids redundant API calls for the same video
- **Profile Database** - SQLite storage with search and history
- **PDF Reports** - Professional formatted export
- **NCI Methodology** - Chase Hughes behavioral science framework
- **Deception Conflict Matrix** - Reconciles contradicting signals across modalities

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "FFmpeg not found" | Run `setup.bat` again, or manually download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and place `ffmpeg.exe` in the project folder |
| "API key invalid" | Verify your key at [openrouter.ai/keys](https://openrouter.ai/keys) and re-enter |
| "Video too long/large" | Videos must be 10-300 seconds, under 100MB |
| Analysis fails midway | Check OpenRouter credits; some models have usage limits |
| Charts not showing | Ensure `plotly` installed: `pip install plotly` |

## License

MIT

## Disclaimer

For educational and research purposes only. AI-generated psychological profiles are speculative assessments, not clinical diagnoses. Do not use for employment, legal, or medical decisions.
