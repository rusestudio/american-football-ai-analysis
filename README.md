# 🏈 American Football AI Analysis

An AI-powered system for analyzing American football games from drone aerial footage. Uses computer vision and large language models to generate actionable coaching reports.

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This system processes drone video footage of football games and provides:

- **Player Tracking**: Real-time tracking of all players and the ball using YOLO + ByteTrack
- **Team Assignment**: Automatic red vs yellow team classification using HSV color detection
- **Play Segmentation**: Detects individual plays (huddle → pre-snap → snap → tackle)
- **Formation Detection**: Classifies offensive and defensive formations pre-snap
- **AI Tactical Analysis**: Claude-powered coaching insights and recommendations
- **Annotated Output**: Rendered video with player IDs, teams, formations, and play info

## quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Environment Variables

Create a `.env` file with your API key:

```bash
# Required for AI tactical analysis (optional for basic tracking)
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Add Your Model Weights

Place your trained YOLO weights in the models directory:

```bash
# Option A: Use custom model (recommended for your footage)
mkdir -p models
cp your-trained-model.pt models/best.pt

# Option B: Use generic YOLO (lower accuracy for players)
# Download yolov8x.pt from ultralytics.com
```

### 4. Add Your Video

```bash
# Option A: Rename your video (recommended)
cp your-game-footage.mp4 input_videos/game_footage.mp4

# Option B: Update INPUT_VIDEO in main.py to point to your file
```

### 5. Calibrate (One-Time Setup)

```bash
python calibrate_drone.py
```

Follow the interactive tool to click on two yard-line markings. Enter the real-world yard distance between them. Copy the `PIXELS_PER_YARD` value into `main.py`.

### 6. Run Analysis

```bash
python main.py
```

Output:
- Annotated video: `output-videos/analyzed_output.avi`
- Coaching report: `reports/coaching_report_YYYY-MM-DD_HH-MM.md`

## How It Works

### Pipeline

```
 ┌─────────────────┐
 │  1. Load Video  │
 └────────┬────────┘
          ▼
 ┌─────────────────────────┐
 │  2. YOLO + ByteTrack   │  ← Detect players, referees, ball
 │     Object Tracking   │
 └────────┬────────────────┘
          ▼
 ┌─────────────────────────┐
 │  3. Camera Movement      │  ← Correct for drone drift
 │     Correction          │
 └────────┬────────────────┘
          ▼
 ┌─────────────────────────┐
 │  4. View Transform      │  ← Convert pixels → yards
 └────────┬────────────────┘
          ▼
 ┌─────────────────────────┐
 │  5. Team Assignment     │  ← Red (0) vs Yellow (1)
 └────────┬────────────────┘
          ▼
 ┌─────────────────────────┐
 │  6. Ball Possession    │  ← Who has the ball?
 └────────┬────────────────┘
          ▼
 ┌─────────────────────────┐
 │  7. Speed & Distance   │  ← Calculate player speeds
 └────────┬────────────────┘
          ▼
 ┌─────────────────────────┐
 │  8. Play Segmentation │  ← Detect individual plays
 └────────┬────────────────┘
          ▼
 ┌─────────────────────────┐
 │  9. Formation Detect   │  ← Classify pre-snap formations
 └────────┬────────────────┘
          ▼
 ┌─────────────────────────┐
 │ 10. Claude AI Analysis│  ← Tactical commentary
 └────────┬────────────────┘
          ▼
 ┌─────────────────────────┐
 │ 11. Generate Report   │  ← Markdown coaching report
 └─────────────────────────┘
```

### Key Components

| File | Purpose |
|------|---------|
| `main.py` | Main pipeline orchestrator |
| `tracker.py` | YOLO + ByteTrack object detection |
| `team_assigner.py` | HSV-based team classification |
| `play_segmenter.py` | Play boundaries from motion |
| `formation_detector.py` | Offense/defense formation classification |
| `tactician.py` | Claude AI tactical analysis |
| `calibrate_drone.py` | Interactive calibration tool |

### Formation Detection

**Offense Examples:**
- Shotgun, 11 Personnel, Trips Right
- Under Center, 21 Personnel, Twins Left
- Empty Backfield, 5-Wide

**Defense Examples:**
- 4-3, Cover-2 (Two-High)
- 3-4, Cover-1 (Man)
- Nickel (4-2-5), Cover-3

## Configuration

### main.py Settings

```python
# Input/Output
INPUT_VIDEO  = "input_videos/game_footage.mp4"
OUTPUT_VIDEO = "output-videos/analyzed_output.avi"
MODEL_PATH   = "models/best.pt"

# Calibration (from calibrate_drone.py)
PIXELS_PER_YARD = 18.0

# Play detection sensitivity
HUDDLE_STILLNESS_SECONDS = 2.0

# AI analysis toggle
USE_CLAUDE_ANALYSIS = True

# Team labels
YOUR_TEAM_NAME = "Red Team"
OPP_TEAM_NAME  = "Yellow Team"
```

### team_assigner.py HSV Ranges

The default HSV ranges are tuned for red jerseys (Team 0) vs yellow/gold jerseys (Team 1). Edit these values if your footage has different jersey colors:

```python
# Red jerseys (wraps around 0/180 in HSV)
RED_HSV_LOW1  = np.array([0,   80,  60],  dtype=np.uint8)
RED_HSV_HIGH1 = np.array([15, 255, 255], dtype=np.uint8)
RED_HSV_LOW2  = np.array([160, 80,  60],  dtype=np.uint8)
RED_HSV_HIGH2 = np.array([180, 255, 255], dtype=np.uint8)

# Yellow/Gold jerseys
YEL_HSV_LOW  = np.array([18, 80, 100], dtype=np.uint8)
YEL_HSV_HIGH = np.array([40, 255, 255], dtype=np.uint8)
```

## Output Examples

### Annotated Video

The output video includes:
- **Player IDs** above each player
- **Team colors** (Red/Yellow)
- **Ball carrier** indicator (orange circle)
- **Team possession bar** (top-left)
- **Formation overlay** (top-right)
- **Play info** (yards gained)

### Coaching Report

```markdown
# 🏈 Coaching Analysis Report
**Generated:** 2025-09-27_17-30
**Powered by:** Claude AI + Drone Vision Analysis

---

## Executive Summary

[AI-generated game summary]

---

## Game Statistics

| Metric | Value |
|--------|-------|
| Total Plays Analyzed | 12 |
| Total Yards | 87 |
| Avg Yards/Play | 7.2 |
| Most Used Formation | Shotgun, 11 Personnel |

---

## Play-by-Play Breakdown

### Play 1
**Offense:** Shotgun, 11 Personnel, Trips Right
**Defense:** 4-3, Cover-2 (Two-High)
**Result:** +8 yards gain

Formation Read: The offense is spread out with three receivers to the right...
[Claude tactical analysis...]
```

## Requirements

```
# Core CV & tracking
ultralytics>=8.0.0          # YOLO v8/v9/v11
supervision>=0.18.0         # ByteTrack + annotation helpers
opencv-python>=4.8.0        # frame processing, optical flow

# ML & data
numpy>=1.24.0
scikit-learn>=1.3.0         # KMeans for team color assignment
pandas>=2.1.0               # data handling for reports

# Claude AI
anthropic>=0.40.0           # Claude API (vision + text)

# Utilities
python-dotenv>=1.0.0        # loads ANTHROPIC_API_KEY from .env
matplotlib>=3.8.0           # optional: plot heatmaps, speed charts
Pillow>=10.0.0              # image utilities
```

## Project Structure

```
american-football-ai-analysis/
├── main.py                    # Main pipeline
├── tracker.py                 # YOLO + ByteTrack
├── team_assigner.py           # Team classification
├── play_segmenter.py         # Play detection
├── formation_detector.py     # Formation classification
├── tactician.py              # Claude AI analysis
├── calibrate_drone.py        # Calibration tool
├── requirements.txt          # Dependencies
├── .env                     # API keys
├── input_videos/             # Input footage
│   └── game_footage.mp4
├── output-videos/            # Annotated output
│   └── analyzed_output.avi
├── models/                   # YOLO weights
│   └── best.pt
├── reports/                  # Coaching reports
│   └── coaching_report_*.md
└── stubs/                    # Cached tracking data
    ├── tracking_data.pkl
    └── camera_movement.pkl
```

## Troubleshooting

### No tracking data detected
- Check that your YOLO model is trained to detect players (class 0) and sports balls (class 32)
- Adjust `conf` threshold in `tracker.py`

### Teams not being classified correctly
- Run the calibrate tool to adjust HSV ranges for your specific footage
- Edit HSV ranges in `team_assigner.py`

### "Cannot find ball"
- Ball detection requires COCO class 32 (sports ball) in your YOLO model
- Check that your model detects the ball

### Claude analysis fails
- Verify `ANTHROPIC_API_KEY` is set in `.env`
- Check your API credits/limit

### Camera drift in output
- Run `calibrate_drone.py` to get accurate PIXELS_PER_YARD
- Camera movement correction is in `main.py` step 3

## Customization

### Adding New Formations

Edit `formation_detector.py` to add your own formation logic:

```python
def _build_offense_string(self, ...):
    # Add your custom formation classification
    if custom_condition:
        return "Custom Formation"
```

### Custom Player Roles

Add new position labels in `formation_detector.py`:

```python
# Update position labels
pos_map[player_id] = "CUSTOM_POSITION"
```

### Alternative Tracking Models

Replace the YOLO model in `main.py`:

```bash
# Use a different YOLO variant
MODEL_PATH = "yolov8n.pt"  # nano (faster)
MODEL_PATH = "yolov8x.pt"   # extra-large (more accurate)
```

Or train your own model:

```bash
yolo detect train data=football.yaml model=yolov8x.pt epochs=100
```

## License

MIT License

## Credits

- [Ultralytics YOLO](https://ultralytics.com) — Object detection
- [Supervision](https://supervision.roboflow.com) — Tracking utilities
- [Anthropic Claude](https://anthropic.com) — AI tactical analysis
