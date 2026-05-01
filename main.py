"""
American Football Analysis Pipeline
====================================
Tuned for: 2025-09-27-17-17-41-376.mp4
  - Drone footage, 1280x720, 30fps, ~45° elevated angle
  - Your team: RED (Team 0)
  - Opponent:  YELLOW (Team 1)
  - Active field zone: right side of frame (x 600-1280, y 200-560)

Usage:
  1. Copy your video to input_videos/game_footage.mp4
     (or change INPUT_VIDEO below)
  2. Add ANTHROPIC_API_KEY to .env
  3. Put your YOLO weights in models/best.pt
  4. Run: python main.py
"""

import cv2
import numpy as np
from pathlib import Path

from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from play_segmenter.play_segmenter import PlaySegmenter
from formation_detector.formation_detector import FormationDetector
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedAndDistanceEstimator
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from claude_tactician.tactician import ClaudeTactician
from utils.video_utils import read_video, save_video
from utils.bbox_utils import get_center_of_bbox
from utils.draw_utils import (
    draw_player_annotation,
    draw_team_stats_overlay,
    draw_formation_overlay,
    draw_play_info_overlay,
)

from dotenv import load_dotenv
import os

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8x.pt")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Change INPUT_VIDEO to point to your drone file
INPUT_VIDEO  = "input_videos/game_footage.mp4"
OUTPUT_VIDEO = "output_videos/analyzed_output.avi"
MODEL_PATH   = "models/best.pt"
REPORT_DIR   = "reports/"
STUB_PATH    = "stubs/tracking_data.pkl"

# Drone calibration for your footage
# Run calibrate_drone.py first, then paste value here
# Rough estimate for 1280x720 drone at ~50ft: ~18px per yard
PIXELS_PER_YARD = 18.0

# Play detection sensitivity
# 2 seconds of stillness = huddle / dead ball
HUDDLE_STILLNESS_SECONDS = 2.0

# Set False to skip Claude API (no ANTHROPIC_API_KEY needed for testing)
USE_CLAUDE_ANALYSIS = True

# Your team color label (for report readability)
YOUR_TEAM_NAME = "Red Team"
OPP_TEAM_NAME  = "Yellow Team"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  AMERICAN FOOTBALL ANALYSIS")
    print(f"  {YOUR_TEAM_NAME} vs {OPP_TEAM_NAME}")
    print("=" * 60)

    # 1. Load video
    print("\n[1/7] Loading video...")
    video_frames = read_video(INPUT_VIDEO)
    fps = _get_fps(INPUT_VIDEO)
    print(f"      {len(video_frames)} frames @ {fps:.1f} fps  "
          f"({len(video_frames)/fps:.1f}s)")

    # 2. YOLO tracking
    print("\n[2/7] Running YOLO tracking...")
    tracker = Tracker(MODEL_PATH)
    tracks  = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path=STUB_PATH
    )
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    print(f"      Tracking complete. Players tracked: "
          f"{len(tracks['players'][0])} in frame 0")

    # 3. Camera movement correction (drone drift)
    print("\n[3/7] Correcting camera movement...")
    cam = CameraMovementEstimator(video_frames[0])
    cam_movement = cam.get_camera_movement(
        video_frames, read_from_stub=True,
        stub_path="stubs/camera_movement.pkl"
    )
    cam.add_adjust_positions_to_tracks(tracks, cam_movement)

    # 4. Perspective transform → yards
    print("\n[4/7] Transforming to real-world coordinates...")
    vt = ViewTransformer(pixels_per_yard=PIXELS_PER_YARD)
    vt.add_transformed_position_to_tracks(tracks)

    # 5. Team assignment
    print("\n[5/7] Assigning teams (red vs yellow)...")
    ta = TeamAssigner()
    ta.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for pid, data in player_track.items():
            team = ta.get_player_team(video_frames[frame_num], data["bbox"], pid)
            tracks["players"][frame_num][pid]["team"]       = team
            tracks["players"][frame_num][pid]["team_color"] = ta.team_colors.get(
                team, (200, 200, 200)
            )

    # 6. Ball possession
    ba = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox")
        carrier   = ba.assign_ball_to_player(player_track, ball_bbox)
        if carrier != -1:
            tracks["players"][frame_num][carrier]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][carrier].get("team", -1)
            )
        else:
            team_ball_control.append(
                team_ball_control[-1] if team_ball_control else -1
            )

    # 7. Speed & distance
    SpeedAndDistanceEstimator(fps=fps).add_speed_and_distance_to_tracks(tracks)

    # 8. Play segmentation
    print("\n[6/7] Segmenting plays...")
    seg   = PlaySegmenter(fps=fps, huddle_stillness_seconds=HUDDLE_STILLNESS_SECONDS)
    plays = seg.segment_plays(tracks, video_frames)
    print(f"      {len(plays)} plays detected")
    for i, p in enumerate(plays):
        print(f"        Play {i+1}: frame {p['start_frame']}–{p['end_frame']} "
              f"({p['duration_s']:.1f}s)")

    # 9. Formation detection + Claude analysis
    print("\n[7/7] Analyzing formations + generating coaching report...")
    fd        = FormationDetector()
    tactician = ClaudeTactician() if USE_CLAUDE_ANALYSIS else None

    play_analyses = []
    for i, play in enumerate(plays):
        print(f"      Analyzing play {i+1}/{len(plays)}...", end=" ", flush=True)

        pre_snap_frame  = video_frames[play["start_frame"]]
        pre_snap_tracks = tracks["players"][play["start_frame"]]

        off_form, def_form, pos_map = fd.detect_formations(
            pre_snap_tracks, pre_snap_frame
        )

        yards = seg.compute_yards_gained(
            tracks, play["start_frame"], play["end_frame"]
        )

        data = {
            "play_number":       i + 1,
            "start_frame":       play["start_frame"],
            "end_frame":         play["end_frame"],
            "offense_formation": off_form,
            "defense_formation": def_form,
            "position_map":      pos_map,
            "yards_gained":      yards,
            "player_speeds":     play.get("player_speeds", {}),
            "pre_snap_frame":    pre_snap_frame,
        }

        if tactician:
            data["claude_analysis"] = tactician.analyze_play(data)
            print("✓ Claude analyzed")
        else:
            data["claude_analysis"] = "Claude analysis disabled."
            print("✓ (no Claude)")

        play_analyses.append(data)

    # 10. Save coaching report
    if tactician and play_analyses:
        path = tactician.generate_full_report(
            play_analyses, output_dir=REPORT_DIR
        )
        print(f"\n✅ Coaching report → {path}")

    # 11. Render output video
    print("\nRendering annotated video...")
    out_frames = _annotate(video_frames, tracks, team_ball_control, play_analyses)
    save_video(out_frames, OUTPUT_VIDEO, fps=fps)
    print(f"✅ Output video → {OUTPUT_VIDEO}")
    print("\n🏈 Done!")


# ── ANNOTATION ────────────────────────────────────────────────────────────────

def _annotate(video_frames, tracks, team_ball_control, play_analyses):
    frame_to_play = {}
    for a in play_analyses:
        for f in range(a["start_frame"], a["end_frame"] + 1):
            frame_to_play[f] = a

    output = []
    for fn, frame in enumerate(video_frames):
        frame = frame.copy()

        for pid, player in tracks["players"][fn].items():
            frame = draw_player_annotation(
                frame,
                player["bbox"],
                pid,
                player.get("team_color", (180, 180, 180)),
                player.get("has_ball", False),
                player.get("role", ""),
                player.get("speed", 0.0),
            )

        for _, ball in tracks["ball"][fn].items():
            cx, cy = get_center_of_bbox(ball["bbox"])
            cv2.circle(frame, (cx, cy), 8, (0, 200, 255), -1)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 0),     2)

        frame = draw_team_stats_overlay(frame, team_ball_control[:fn+1])

        if fn in frame_to_play:
            a = frame_to_play[fn]
            frame = draw_formation_overlay(
                frame, a["offense_formation"], a["defense_formation"]
            )
            frame = draw_play_info_overlay(
                frame, a["play_number"], a.get("yards_gained")
            )

        output.append(frame)

    return output


def _get_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0


if __name__ == "__main__":
    main()
