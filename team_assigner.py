"""
team_assigner.py
=================
Tuned for YOUR footage:
  - Your team (Team 0): RED/CRIMSON jerseys + black pants
  - Opponent (Team 1): YELLOW/GOLD jerseys + white helmets
  - Referees: black & white stripes → excluded (team = -1)

Video: 2025-09-27-17-17-41-376.mp4 (drone, 1280x720, 30fps)

Color detection uses HSV ranges confirmed from your footage.
Falls back to KMeans if HSV gives ambiguous result.
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from typing import Dict, Tuple, Optional


# ── HSV ranges confirmed from your footage ──────────────────────────────────
# Red/Crimson: hue wraps around 0/180 in HSV
RED_HSV_LOW1  = np.array([0,   80,  60],  dtype=np.uint8)
RED_HSV_HIGH1 = np.array([15,  255, 255], dtype=np.uint8)
RED_HSV_LOW2  = np.array([160, 80,  60],  dtype=np.uint8)
RED_HSV_HIGH2 = np.array([180, 255, 255], dtype=np.uint8)

# Yellow/Gold
YEL_HSV_LOW  = np.array([18, 80,  100], dtype=np.uint8)
YEL_HSV_HIGH = np.array([40, 255, 255], dtype=np.uint8)

# Team IDs
TEAM_RED    = 0   # your team
TEAM_YELLOW = 1   # opponent
TEAM_REF    = -1  # referee / staff — excluded from analysis


class TeamAssigner:
    """
    Assigns each tracked player to Team 0 (red) or Team 1 (yellow).

    Strategy (two-pass):
      1. HSV range check — fast, reliable for your footage
      2. KMeans fallback — for edge cases (shadows, partial occlusion)
    """

    def __init__(self):
        self.team_colors: Dict[int, Tuple[int, int, int]] = {
            TEAM_RED:    (60, 60, 220),    # BGR: red display color
            TEAM_YELLOW: (0,  200, 255),   # BGR: yellow display color
        }
        self.player_team_dict: Dict[int, int] = {}
        self._kmeans: Optional[KMeans] = None

    # ── Public ─────────────────────────────────────────────────────────────────

    def assign_team_color(
        self,
        frame: np.ndarray,
        player_detections: Dict,
    ) -> None:
        """
        Fit a KMeans backup model from frame 0.
        The primary method is HSV-based, but KMeans is kept
        as fallback for ambiguous detections.
        """
        player_colors = []
        for _, data in player_detections.items():
            color = self._sample_player_color(frame, data["bbox"])
            if color is not None:
                player_colors.append(color)

        if len(player_colors) >= 4:
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            km.fit(player_colors)
            self._kmeans = km

    def get_player_team(
        self,
        frame: np.ndarray,
        player_bbox,
        player_id: int,
    ) -> int:
        """
        Returns TEAM_RED (0), TEAM_YELLOW (1), or TEAM_REF (-1).
        Result is cached per player_id to avoid flicker.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        team = self._classify_by_hsv(frame, player_bbox)

        # Cache result (only cache confirmed team members, not refs)
        if team != TEAM_REF:
            self.player_team_dict[player_id] = team

        return team

    # ── HSV Classification ────────────────────────────────────────────────────

    def _classify_by_hsv(self, frame: np.ndarray, bbox) -> int:
        """
        Primary classifier using HSV color ranges from your footage.
        Counts red vs yellow pixels in the player bounding box.
        Whoever has more pixels wins.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(frame.shape[1]-1, x2)
        y2 = min(frame.shape[0]-1, y2)

        if x2 <= x1 or y2 <= y1:
            return TEAM_RED  # safe default

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return TEAM_RED

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Count red pixels (two HSV ranges because red wraps)
        red_mask1 = cv2.inRange(hsv, RED_HSV_LOW1, RED_HSV_HIGH1)
        red_mask2 = cv2.inRange(hsv, RED_HSV_LOW2, RED_HSV_HIGH2)
        red_count = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)

        # Count yellow pixels
        yel_mask  = cv2.inRange(hsv, YEL_HSV_LOW, YEL_HSV_HIGH)
        yel_count = cv2.countNonZero(yel_mask)

        total_pixels = roi.shape[0] * roi.shape[1]

        # Referee check: very low saturation + striped pattern
        if self._is_referee(hsv, red_count, yel_count, total_pixels):
            return TEAM_REF

        # Need at least 3% colored pixels to make a call
        threshold = total_pixels * 0.03
        if red_count < threshold and yel_count < threshold:
            # Fall back to KMeans
            return self._classify_by_kmeans(roi)

        return TEAM_RED if red_count >= yel_count else TEAM_YELLOW

    def _classify_by_kmeans(self, roi: np.ndarray) -> int:
        """KMeans fallback for low-saturation or shadowed players."""
        if self._kmeans is None:
            return TEAM_RED

        color = self._dominant_color(roi)
        if color is None:
            return TEAM_RED

        team = int(self._kmeans.predict([color])[0])
        return team

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sample_player_color(
        self, frame: np.ndarray, bbox
    ) -> Optional[np.ndarray]:
        """Return mean BGR color of player ROI, excluding grass."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(frame.shape[1]-1, x2)
        y2 = min(frame.shape[0]-1, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame[y1:y2, x1:x2]
        return self._dominant_color(roi)

    @staticmethod
    def _dominant_color(roi: np.ndarray) -> Optional[np.ndarray]:
        """Return dominant non-grass BGR color in region."""
        pixels = roi.reshape(-1, 3).astype(np.float32)
        # Remove grass-green pixels
        non_grass = [p for p in pixels
                     if not (p[1] > p[0]*1.15 and p[1] > p[2]*1.1 and p[1] > 70)]
        if len(non_grass) < 5:
            return None
        return np.mean(non_grass, axis=0)

    @staticmethod
    def _is_referee(
        hsv_roi: np.ndarray,
        red_count: int,
        yel_count: int,
        total: int,
    ) -> bool:
        """
        Referees wear black & white stripes.
        Detect by very low saturation across the ROI.
        """
        mean_sat = float(np.mean(hsv_roi[:, :, 1]))
        # If avg saturation < 40 and neither team color dominates → referee
        return mean_sat < 40 and red_count < total*0.02 and yel_count < total*0.02
