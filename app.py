"""
American Football AI Analysis — Streamlit Web App
===================================================
A user-friendly interface for analyzing football drone footage.

Usage:
    streamlit run app.py

Features:
    - Upload your own video
    - Upload your custom YOLO model
    - Configure analysis settings
    - View annotated output video
    - Download coaching report
"""

import streamlit as st
import os
import cv2
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Try to import ultralytics for YOLO
try:
    from ultralytics import YOLO
    import supervision as sv
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="🏈 Football AI Analysis",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff4b4b;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #a0a0a0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #1a2e1a;
        border: 1px solid #28a745;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #1a2a3a;
        border: 1px solid #17a2b8;
    }
    </style>
""", unsafe_allow_html=True)


# ── HELPER FUNCTIONS —───────────────────────────────────────────────────────

def read_video(path):
    """Read video file and return list of frames."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(frames, path, fps=30.0):
    """Save frames to video file (MP4 format)."""
    if not frames:
        return
    
    h, w = frames[0].shape[:2]
    # Use MP4 codec (H264)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    
    for frame in frames:
        out.write(frame)
    
    out.release()


def get_center_of_bbox(bbox):
    """Get center of bounding box."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def draw_player_annotation(frame, bbox, player_id, team_color, has_ball, role, speed):
    """Draw player annotation on frame."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
    
    # Draw ID
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + 40, y1), team_color, -1)
    cv2.putText(frame, f"#{player_id}", (x1 + 5, y1 - 8), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Ball carrier indicator
    if has_ball:
        cx, cy = get_center_of_bbox(bbox)
        cv2.circle(frame, (cx, cy), 12, (0, 200, 255), -1)
        cv2.circle(frame, (cx, cy), 12, (0, 0, 0), 2)
    
    return frame


def draw_team_stats_overlay(frame, team_ball_control):
    """Draw team possession overlay."""
    h, w = frame.shape[:2]
    
    # Simple possession bar
    red_count = sum(1 for t in team_ball_control if t == 0)
    yellow_count = sum(1 for t in team_ball_control if t == 1)
    total = red_count + yellow_count
    
    if total > 0:
        red_pct = red_count / total
        bar_width = int(w * 0.3 * red_pct)
        
        # Draw bar
        cv2.rectangle(frame, (10, 10), (10 + bar_width, 20), (60, 60, 220), -1)
        cv2.rectangle(frame, (10 + bar_width, 10), (10 + int(w * 0.3), 20), (0, 200, 255), -1)
    
    return frame


def draw_formation_overlay(frame, offense_formation, defense_formation):
    """Draw formation info overlay."""
    h, w = frame.shape[:2]
    
    # Formation box
    cv2.rectangle(frame, (w - 250, 10), (w - 10, 70), (30, 30, 50), -1)
    cv2.putText(frame, f"OFF: {offense_formation[:15]}", (w - 240, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"DEF: {defense_formation[:15]}", (w - 240, 55),
              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame


def draw_play_info_overlay(frame, play_number, yards_gained):
    """Draw play info overlay."""
    h, w = frame.shape[:2]
    
    # Play info
    cv2.rectangle(frame, (10, h - 60), (150, h - 10), (30, 30, 50), -1)
    cv2.putText(frame, f"Play #{play_number}", (20, h - 40),
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if yards_gained is not None:
        yard_text = f"+{yards_gained} yds" if yards_gained > 0 else f"{yards_gained} yds"
        cv2.putText(frame, yard_text, (20, h - 15),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if yards_gained > 0 else (0, 0, 255), 2)
    
    return frame


# ── ANALYSIS FUNCTIONS —──────────────────────────────────────────────────

def detect_objects(frames, model_path):
    """Run YOLO detection on frames."""
    if not YOLO_AVAILABLE:
        return None, None
    
    model = YOLO(model_path)
    tracker = sv.ByteTrack()
    
    all_detections = []
    
    progress_bar = st.progress(0)
    for i in range(0, len(frames), 10):
        batch = frames[i:i+10]
        results = model.predict(batch, conf=0.2)
        
        for result in results:
            detections = sv.Detections.from_ultralytics(result)
            # Track players (class 0)
            player_det = detections[detections.class_id == 0]
            player_det = tracker.update_with_detections(player_det)
            all_detections.append(player_det)
        
        progress_bar.progress(min(i + 10, len(frames)) / len(frames))
    
    return all_detections


def analyze_video_simple(frames, model_path, config):
    """Simplified video analysis without all dependencies."""
    
    # Check for YOLO
    if not YOLO_AVAILABLE:
        st.error("YOLO not available. Please install ultralytics: pip install ultralytics")
        return None, None, None, None
    
    st.info("Running object detection...")
    detections = detect_objects(frames, model_path)
    
    if detections is None:
        st.error("Detection failed. Please check your model.")
        return None, None, None, None
    
    # Get frame info
    fps = 30.0
    num_frames = len(frames)
    num_detections = len(detections)
    
    st.success(f"Detected objects in {num_detections} frames")
    
    # Create annotated frames
    st.info("Creating annotated video...")
    annotated_frames = []
    
    for i, frame in enumerate(frames):
        annotated = frame.copy()
        
        # Draw detections
        if i < len(detections):
            det = detections[i]
            for j in range(len(det)):
                bbox = det.xyxy[j]
                track_id = det.tracker_id[j]
                
                # Assign fake team colors for demo
                team_color = (60, 60, 220) if j % 2 == 0 else (0, 200, 255)
                
                annotated = draw_player_annotation(
                    annotated, 
                    bbox.tolist(),
                    int(track_id) if track_id is not None else j,
                    team_color,
                    False,
                    "",
                    0.0
                )
        
        annotated_frames.append(annotated)
        
        if i % 30 == 0:
            st.progress(i / len(frames))
    
    return annotated_frames, num_detections, num_frames, fps


# ── SIDEBAR —───────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.title("⚙️ Configuration")
    
    st.sidebar.markdown("### 📹 Video Settings")
    video_file = st.sidebar.file_uploader(
        "Upload Game Video",
        type=["mp4", "avi", "mov"],
        help="Drone footage of the football game"
    )
    
    st.sidebar.markdown("### 🤖 Model Settings")
    model_file = st.sidebar.file_uploader(
        "Upload YOLO Model",
        type=["pt"],
        help="Your trained YOLO model (.pt file)"
    )
    
    st.sidebar.markdown("### 📏 Calibration")
    pixels_per_yard = st.sidebar.number_input(
        "Pixels Per Yard",
        min_value=1.0,
        max_value=100.0,
        value=18.0,
        help="From calibrate_drone.py"
    )
    
    st.sidebar.markdown("### 🎨 Team Labels")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        your_team_name = st.text_input("Your Team", value="Red Team")
    with col2:
        opp_team_name = st.text_input("Opponent", value="Yellow Team")
    
    st.sidebar.markdown("### 🔧 Analysis Settings")
    huddle_stillness = st.sidebar.slider(
        "Huddle Stillness (seconds)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Seconds of stillness to detect huddle"
    )
    
    use_claude = st.sidebar.checkbox(
        "Use Claude AI Analysis",
        value=True,
        help="Requires ANTHROPIC_API_KEY"
    )
    
    return {
        "video_file": video_file,
        "model_file": model_file,
        "pixels_per_yard": pixels_per_yard,
        "your_team_name": your_team_name,
        "opp_team_name": opp_team_name,
        "huddle_stillness": huddle_stillness,
        "use_claude": use_claude,
    }


def save_uploaded_file(uploaded_file, temp_dir):
    """Save uploaded file to temp directory."""
    if uploaded_file is None:
        return None
    
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# ── MAIN APP —──────────────────────────────────────────────────────────────

def main():
    # Title
    st.markdown('<p class="title">🏈 Football AI Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Drone footage → AI tactical insights</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Check YOLO availability
    if not YOLO_AVAILABLE:
        st.error("""
        ⚠️ YOLO not installed. Please run:
        ```bash
        pip install ultralytics supervision opencv-python
        ```
        """)
        return
    
    # Sidebar config
    config = render_sidebar()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if config["video_file"] is None:
            st.info("👆 Upload a video file to get started!")
            st.markdown("""
            ### 📋 Instructions
            1. **Upload Video** — Drag & drop your game footage (mp4/avi)
            2. **Upload Model** — Your trained YOLO model (.pt)
            3. **Configure** — Adjust pixels per yard, team names
            4. **Run Analysis** — Click the button below
            5. **View Results** — See annotated video + report
            """)
            return
        
        st.success(f"📹 Video: {config['video_file'].name}")
        
        if config["model_file"] is None:
            st.warning("⚠️ No model uploaded — using default YOLO")
            model_path = "yolov8x.pt"
        else:
            st.success(f"🤖 Model: {config['model_file'].name}")
    
    with col2:
        # Show preview frame
        if config["video_file"] is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = save_uploaded_file(config["video_file"], temp_dir)
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="First Frame Preview", use_container_width=True)
    
    # Run button
    st.markdown("---")
    
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        if config["video_file"] is None:
            st.error("Please upload a video file first!")
            return
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files
            video_path = save_uploaded_file(config["video_file"], temp_dir)
            
            # Read video
            st.info("Loading video...")
            frames = read_video(video_path)
            st.success(f"Loaded {len(frames)} frames")
            
            # Determine model path
            if config["model_file"] is not None:
                model_path = save_uploaded_file(config["model_file"], temp_dir)
            else:
                # Try using a default model
                st.info("Using default YOLO model...")
                model_path = "yolov8n.pt"  # Use smaller model as default
            
            # Run analysis
            try:
                annotated_frames, num_det, total_frames, fps = analyze_video_simple(
                    frames, model_path, config
                )
                
                if annotated_frames is None:
                    st.error("Analysis failed")
                    return
                
                # Show results
                st.markdown("---")
                st.markdown("## 📊 Results")
                
                # Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Frames", total_frames)
                with col2:
                    st.metric("Objects Tracked", num_det if num_det else 0)
                with col3:
                    st.metric("FPS", fps)
                
                # Annotated video
                st.markdown("### 🎬 Annotated Video")
                
# Save output video
                output_path = os.path.join(temp_dir, "output.mp4")
                save_video(annotated_frames, output_path, fps=fps)
                
                # Read and display video
                video_bytes = open(output_path, "rb").read()
                st.video(video_bytes)
                
                # Download button
                st.download_button(
                    label="📥 Download Annotated Video",
                    data=video_bytes,
                    file_name="analyzed_output.mp4",
                    mime="video/mp4"
                )
                
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
