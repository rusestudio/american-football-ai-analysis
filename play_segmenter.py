"""
play_segmenter.py
==================
Detects individual plays from a continuous game video.

American football has discrete plays separated by huddles.
The key insight for top-down drone footage:

  HUDDLE  → all players clustered in small area, low movement
  PRE-SNAP → players spread out and hold position (≥2s stillness)
  SNAP    → sudden collective movement starts
  PLAY    → players moving
  DEAD BALL → collective stillness returns (tackle, out-of-bounds)

We use collective player velocity to detect transitions.
"""

import numpy as np
from typing import List, Dict, Any, Tuple


class PlaySegmenter:
    """
    Segments a continuous football video into individual plays.

    Parameters
    ----------
    fps : float
        Frames per second of the drone video.
    huddle_stillness_seconds : float
        How many seconds of low collective motion = dead ball / huddle.
    motion_threshold : float
        Average player speed (yards/frame) above which a frame is "live".
    min_play_seconds : float
        Discard detected plays shorter than this (usually false positives).
    max_play_seconds : float
        Cap play length at this many seconds (handles edge cases).
    """

    def __init__(
        self,
        fps: float = 30.0,
        huddle_stillness_seconds: float = 2.0,
        motion_threshold: float = 0.15,   # yards per frame
        min_play_seconds: float = 2.0,
        max_play_seconds: float = 15.0,
    ):
        self.fps                      = fps
        self.huddle_stillness_frames  = int(huddle_stillness_seconds * fps)
        self.motion_threshold         = motion_threshold
        self.min_play_frames          = int(min_play_seconds * fps)
        self.max_play_frames          = int(max_play_seconds * fps)

    # ── Public ─────────────────────────────────────────────────────────────────

    def segment_plays(
        self,
        tracks: Dict[str, List[Dict]],
        video_frames: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Main entry point.

        Returns a list of play dicts:
          {
            "start_frame": int,   # first frame of play (snap)
            "end_frame":   int,   # last frame of play (tackle/OOB)
            "duration_s":  float, # play duration in seconds
            "player_speeds": {player_id: avg_speed_yards_per_s},
          }
        """
        motion_signal = self._compute_motion_signal(tracks)
        raw_plays     = self._detect_play_boundaries(motion_signal)
        plays         = self._filter_plays(raw_plays, tracks)
        return plays

    def compute_yards_gained(
        self,
        tracks: Dict[str, List[Dict]],
        start_frame: int,
        end_frame: int,
    ) -> float:
        """
        Estimate yards gained on a play.

        Strategy: find the ball-carrier at snap and at play end.
        Yards gained = difference in their field position (y-axis
        in a top-down view, since the field runs vertically).

        Returns None if ball position is not reliably tracked.
        """
        ball_start = self._get_ball_position(tracks, start_frame)
        ball_end   = self._get_ball_position(tracks, end_frame)

        if ball_start is None or ball_end is None:
            return None

        # In top-down view y-axis = field length direction.
        # Positive = toward end zone (we assume offense goes up).
        yards = ball_end[1] - ball_start[1]
        return round(float(yards), 1)

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _compute_motion_signal(
        self, tracks: Dict[str, List[Dict]]
    ) -> np.ndarray:
        """
        Compute per-frame collective player motion.

        Returns array of shape (n_frames,) where each value is the
        average speed (yards/frame) across all detected players.
        """
        player_tracks = tracks["players"]
        n_frames      = len(player_tracks)
        motion        = np.zeros(n_frames)

        for frame_num in range(1, n_frames):
            prev_frame = player_tracks[frame_num - 1]
            curr_frame = player_tracks[frame_num]

            speeds = []
            for player_id, curr_data in curr_frame.items():
                if player_id not in prev_frame:
                    continue

                curr_pos = curr_data.get("position_transformed")
                prev_pos = prev_frame[player_id].get("position_transformed")

                if curr_pos is None or prev_pos is None:
                    # Fall back to pixel position if transform not ready
                    curr_pos = self._bbox_center(curr_data.get("bbox"))
                    prev_pos = self._bbox_center(
                        prev_frame[player_id].get("bbox")
                    )

                if curr_pos and prev_pos:
                    dx = curr_pos[0] - prev_pos[0]
                    dy = curr_pos[1] - prev_pos[1]
                    speeds.append(np.sqrt(dx**2 + dy**2))

            motion[frame_num] = np.mean(speeds) if speeds else 0.0

        return motion

    def _detect_play_boundaries(
        self, motion_signal: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        State machine over the motion signal to find play start/end.

        States:
          DEAD  → signal below threshold for huddle_stillness_frames
          LIVE  → signal above threshold (play in progress)

        Transition DEAD→LIVE = snap (play start)
        Transition LIVE→DEAD = tackle/OOB (play end)
        """
        plays       = []
        state       = "DEAD"
        play_start  = None
        still_count = 0

        for f, speed in enumerate(motion_signal):
            if state == "DEAD":
                if speed > self.motion_threshold:
                    # Snap! Play has started.
                    play_start  = f
                    state       = "LIVE"
                    still_count = 0
            else:  # LIVE
                if speed <= self.motion_threshold:
                    still_count += 1
                    if still_count >= self.huddle_stillness_frames:
                        # Dead ball confirmed
                        play_end = f - self.huddle_stillness_frames
                        plays.append((play_start, play_end))
                        state       = "DEAD"
                        play_start  = None
                        still_count = 0
                else:
                    still_count = 0

        # Close any open play at video end
        if state == "LIVE" and play_start is not None:
            plays.append((play_start, len(motion_signal) - 1))

        return plays

    def _filter_plays(
        self,
        raw_plays: List[Tuple[int, int]],
        tracks: Dict[str, List[Dict]],
    ) -> List[Dict[str, Any]]:
        """Remove plays that are too short/long and enrich with metadata."""
        filtered = []
        for start, end in raw_plays:
            duration_frames = end - start
            if duration_frames < self.min_play_frames:
                continue
            if duration_frames > self.max_play_frames:
                end = start + self.max_play_frames

            duration_s    = duration_frames / self.fps
            player_speeds = self._compute_per_player_speed(tracks, start, end)

            filtered.append({
                "start_frame":   start,
                "end_frame":     end,
                "duration_s":    round(duration_s, 2),
                "player_speeds": player_speeds,
            })

        return filtered

    def _compute_per_player_speed(
        self,
        tracks: Dict[str, List[Dict]],
        start_frame: int,
        end_frame: int,
    ) -> Dict[int, float]:
        """Average speed (yards/s) for each player during the play."""
        player_speeds = {}
        player_tracks = tracks["players"]

        for player_id in player_tracks[start_frame]:
            positions = []
            for f in range(start_frame, min(end_frame + 1, len(player_tracks))):
                if player_id not in player_tracks[f]:
                    continue
                pos = player_tracks[f][player_id].get("position_transformed")
                if pos:
                    positions.append(pos)

            if len(positions) < 2:
                continue

            total_dist = sum(
                np.sqrt(
                    (positions[i][0] - positions[i-1][0])**2 +
                    (positions[i][1] - positions[i-1][1])**2
                )
                for i in range(1, len(positions))
            )
            duration_s = (end_frame - start_frame) / self.fps
            player_speeds[player_id] = round(total_dist / duration_s, 2)

        return player_speeds

    def _get_ball_position(
        self,
        tracks: Dict[str, List[Dict]],
        frame_num: int,
    ):
        """Return ball (x, y) in yards, or None if unavailable."""
        ball_track = tracks["ball"]
        if frame_num >= len(ball_track):
            return None
        for _, ball in ball_track[frame_num].items():
            pos = ball.get("position_transformed")
            if pos:
                return pos
            bbox = ball.get("bbox")
            if bbox:
                return self._bbox_center(bbox)
        return None

    @staticmethod
    def _bbox_center(bbox):
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
