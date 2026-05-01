"""
formation_detector.py
======================
Classifies pre-snap offensive and defensive formations.

Tuned for your drone footage (2025-09-27-17-17-41-376.mp4):
  - 1280x720 resolution, ~45° elevated drone angle
  - Active field zone: x 700-1280, y 250-520 (right side of frame)
  - Red team = your team (offense when you have the ball)
  - Yellow team = opponent

Formation outputs
-----------------
Offense examples:
  "Shotgun, 11 Personnel, Trips Right"
  "Under Center, 21 Personnel, Twins Left"
  "Empty Backfield, 5-Wide"

Defense examples:
  "4-3, Cover-2 (Two-High)"
  "3-4, Cover-1 (Man)"
  "Nickel (4-2-5), Cover-3"
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional


# ── FIELD ZONE (tuned for your Video B drone footage) ───────────────────────
# These define where the actual play happens in the frame.
# Sideline crowds and benches outside this zone are ignored.
FIELD_X1 = 600   # left boundary of active field (pixels)
FIELD_X2 = 1280  # right boundary
FIELD_Y1 = 200   # top boundary (away end zone direction)
FIELD_Y2 = 560   # bottom boundary (your bench side)

# Spatial thresholds (pixels, for 1280x720 ~45° drone view)
# These are tuned from your actual footage measurements
LOS_TOLERANCE_PX      = 30   # within this y-dist of LOS = "on the line"
BACKFIELD_MIN_PX      = 35   # offense: this far behind LOS = backfield
QB_SHOTGUN_MIN_PX     = 55   # QB at least this far back = shotgun
WIDE_SPLIT_MIN_PX     = 160  # at least this far from field center x = split wide
LB_DEPTH_MIN_PX       = 35   # defense: this far behind D-line = linebacker
LB_DEPTH_MAX_PX       = 100  # deeper than this = safety/DB
SAFETY_DEPTH_MIN_PX   = 100  # at least this far behind LOS = safety


class FormationDetector:
    """
    Detects pre-snap formations from player positions.
    Works directly from YOLO tracks — no YOLO needed for the
    formation logic itself, only the (x,y) positions matter.
    """

    def detect_formations(
        self,
        player_tracks: Dict[int, Dict],
        frame: np.ndarray,
    ) -> Tuple[str, str, Dict[int, str]]:
        """
        Main entry point — call this on the first frame of each play.

        Parameters
        ----------
        player_tracks : {player_id: {"bbox": [...], "team": 0|1, ...}}
        frame         : current video frame (for fallback pixel analysis)

        Returns
        -------
        offense_formation : str
        defense_formation : str
        position_map      : {player_id: role_label}
        """
        positions = self._extract_positions(player_tracks)

        if len(positions) < 4:
            return "Unknown (too few players)", "Unknown", {}

        # Filter to active field zone only — ignore bench/sideline
        positions = self._filter_to_field_zone(positions)

        if len(positions) < 4:
            return "Unknown (out of field zone)", "Unknown", {}

        # Find line of scrimmage
        los_y = self._find_los(positions)

        # Split offense / defense by team assignment
        offense_ids, defense_ids = self._split_by_team(
            player_tracks, positions, los_y
        )

        # Classify each side
        off_formation, off_pos_map = self._classify_offense(
            offense_ids, positions, los_y
        )
        def_formation, def_pos_map = self._classify_defense(
            defense_ids, positions, los_y
        )

        position_map = {**off_pos_map, **def_pos_map}
        return off_formation, def_formation, position_map

    # ── Line of Scrimmage ────────────────────────────────────────────────────

    def _find_los(self, positions: Dict) -> float:
        """
        Find the y-coordinate of the LOS.
        = the largest vertical gap between any two adjacent player y-values.
        """
        ys = sorted(p[1] for p in positions.values())
        if len(ys) < 2:
            return float(np.median(ys))

        max_gap, los_y = 0, ys[len(ys)//2]
        for i in range(1, len(ys)):
            gap = ys[i] - ys[i-1]
            if gap > max_gap:
                max_gap = gap
                los_y = (ys[i] + ys[i-1]) / 2.0

        return los_y

    # ── Team Split ───────────────────────────────────────────────────────────

    def _split_by_team(
        self,
        player_tracks: Dict,
        positions: Dict,
        los_y: float,
    ) -> Tuple[List[int], List[int]]:
        """
        Use team labels from TeamAssigner (0=red/your team, 1=yellow/opponent).
        Falls back to spatial split if team info is missing.
        """
        team0 = [pid for pid in positions
                 if player_tracks.get(pid, {}).get("team") == 0]
        team1 = [pid for pid in positions
                 if player_tracks.get(pid, {}).get("team") == 1]

        if team0 and team1:
            # Offense = team with average y BELOW (larger y) the LOS
            avg_y0 = np.mean([positions[p][1] for p in team0])
            avg_y1 = np.mean([positions[p][1] for p in team1])
            # In drone view y increases downward; offense is below LOS
            if avg_y0 > avg_y1:
                return team0, team1   # red = offense
            return team1, team0       # yellow = offense

        # Fallback: players below LOS = offense, above = defense
        below = [pid for pid, (_, y) in positions.items() if y >= los_y]
        above = [pid for pid, (_, y) in positions.items() if y <  los_y]
        return below, above

    # ── Offense ──────────────────────────────────────────────────────────────

    def _classify_offense(
        self,
        player_ids: List[int],
        positions: Dict,
        los_y: float,
    ) -> Tuple[str, Dict[int, str]]:

        if not player_ids:
            return "Unknown Offense", {}

        off_pos   = {pid: positions[pid] for pid in player_ids if pid in positions}
        pos_map   = {}

        # Players on/near the LOS
        on_line   = [pid for pid, (_, y) in off_pos.items()
                     if abs(y - los_y) <= LOS_TOLERANCE_PX]

        # Players in backfield (behind LOS)
        backfield = [pid for pid, (_, y) in off_pos.items()
                     if (y - los_y) > BACKFIELD_MIN_PX]

        # Wide splits: on/near line but far from center x
        center_x  = np.median([x for x, _ in off_pos.values()])
        wide      = [pid for pid in on_line
                     if abs(off_pos[pid][0] - center_x) > WIDE_SPLIT_MIN_PX]
        interior  = [pid for pid in on_line if pid not in wide]

        # Label OL (up to 5 interior players sorted left→right)
        ol_sorted = sorted(interior, key=lambda p: off_pos[p][0])
        ol_labels = ["LT", "LG", "C", "RG", "RT"]
        for i, pid in enumerate(ol_sorted[:5]):
            pos_map[pid] = ol_labels[i]

        # Remaining interior = TEs
        te_count = 0
        for pid in interior[5:]:
            te_count += 1
            pos_map[pid] = f"TE{te_count if te_count > 1 else ''}"

        # Tight ends lined up ON the line but outside OT = also TE
        for pid in wide:
            if abs(off_pos[pid][0] - center_x) < WIDE_SPLIT_MIN_PX + 40:
                te_count += 1
                pos_map[pid] = "TE"
            else:
                # True wide split = WR
                wr_num = sum(1 for v in pos_map.values() if v.startswith("WR")) + 1
                pos_map[pid] = f"WR{wr_num}"

        # Label backfield
        qb_id, rb_ids = self._label_backfield(backfield, off_pos, los_y, pos_map)

        # Build formation string
        wr_count = sum(1 for v in pos_map.values() if v.startswith("WR"))
        rb_count = len(rb_ids)
        formation = self._build_offense_string(
            qb_id, rb_count, te_count, wr_count, off_pos, los_y, center_x
        )

        return formation, pos_map

    def _label_backfield(
        self,
        backfield_ids: List[int],
        positions: Dict,
        los_y: float,
        pos_map: Dict,
    ) -> Tuple[Optional[int], List[int]]:
        if not backfield_ids:
            return None, []

        center_x = np.median([positions[p][0] for p in backfield_ids])

        # QB = deepest player (largest y = furthest behind LOS) OR most central
        def qb_score(pid):
            x, y = positions[pid]
            depth   = y - los_y          # more positive = deeper
            x_dev   = abs(x - center_x)  # less = more central
            return depth * 0.5 + (50 / (x_dev + 1)) * 0.5

        qb_id = max(backfield_ids, key=qb_score)
        pos_map[qb_id] = "QB"

        rb_ids = [p for p in backfield_ids if p != qb_id]
        for i, pid in enumerate(rb_ids):
            pos_map[pid] = "FB" if i == 0 and len(rb_ids) > 1 else "RB"

        return qb_id, rb_ids

    def _build_offense_string(
        self,
        qb_id, rb_count, te_count, wr_count,
        positions, los_y, center_x
    ) -> str:
        parts = []

        # QB alignment
        if qb_id and qb_id in positions:
            depth = positions[qb_id][1] - los_y
            if depth >= QB_SHOTGUN_MIN_PX:
                parts.append("Shotgun")
            elif depth >= 30:
                parts.append("Pistol")
            else:
                parts.append("Under Center")
        else:
            parts.append("Under Center")

        # Personnel package (RB count + TE count)
        if rb_count == 0 and wr_count >= 5:
            parts.append("Empty (5-Wide)")
        else:
            parts.append(f"{rb_count}{te_count} Personnel")

        # Receiver grouping
        if wr_count >= 3:
            # Determine which side has more WRs
            wr_ids = [pid for pid, lbl in
                      {**{qb_id: "QB"}}.items()  # dummy
                      if isinstance(lbl, str) and lbl.startswith("WR")]
            parts.append("Trips")
        elif wr_count == 2:
            parts.append("Twins")

        return ", ".join(parts) if parts else "Pro Set"

    # ── Defense ──────────────────────────────────────────────────────────────

    def _classify_defense(
        self,
        player_ids: List[int],
        positions: Dict,
        los_y: float,
    ) -> Tuple[str, Dict[int, str]]:

        if not player_ids:
            return "Unknown Defense", {}

        def_pos  = {pid: positions[pid] for pid in player_ids if pid in positions}
        pos_map  = {}

        # D-line: players near or on LOS (defense side = above LOS, smaller y)
        on_line  = [pid for pid, (_, y) in def_pos.items()
                    if abs(y - los_y) <= LOS_TOLERANCE_PX + 10]

        # Linebackers: mid-depth behind D-line
        mid      = [pid for pid, (_, y) in def_pos.items()
                    if LB_DEPTH_MIN_PX < (los_y - y) <= LB_DEPTH_MAX_PX]

        # Safeties / DBs: deepest players
        deep     = [pid for pid, (_, y) in def_pos.items()
                    if (los_y - y) > LB_DEPTH_MAX_PX]

        # Label D-line left→right
        dl_sorted = sorted(on_line, key=lambda p: def_pos[p][0])
        dl_labels = self._dl_labels(len(dl_sorted))
        for pid, lbl in zip(dl_sorted, dl_labels):
            pos_map[pid] = lbl

        # Label linebackers
        lb_sorted = sorted(mid, key=lambda p: def_pos[p][0])
        lb_labels = ["WLB", "MLB", "MLB2", "SLB"]
        for i, pid in enumerate(lb_sorted):
            pos_map[pid] = lb_labels[min(i, len(lb_labels)-1)]

        # Label DBs / Safeties
        db_sorted = sorted(deep, key=lambda p: def_pos[p][1])  # sort by depth
        for i, pid in enumerate(db_sorted):
            y = def_pos[pid][1]
            depth = los_y - y
            if depth > SAFETY_DEPTH_MIN_PX:
                pos_map[pid] = "FS" if i == 0 else "SS"
            else:
                cb_num = sum(1 for v in pos_map.values() if v.startswith("CB")) + 1
                pos_map[pid] = f"CB{cb_num}"

        # Build string
        dl_count = len(on_line)
        lb_count = len(mid)
        db_count = len(deep)

        front    = self._front_label(dl_count, lb_count, db_count)
        coverage = self._coverage_label(deep, def_pos, los_y)

        return f"{front}, {coverage}", pos_map

    def _dl_labels(self, n: int) -> List[str]:
        if n == 4: return ["LE", "DT", "DT", "RE"]
        if n == 3: return ["DE", "NT", "DE"]
        if n == 5: return ["LE", "DT", "NT", "DT", "RE"]
        return [f"DL{i+1}" for i in range(n)]

    def _front_label(self, dl: int, lb: int, db: int) -> str:
        if dl == 4 and lb == 3: return "4-3"
        if dl == 3 and lb == 4: return "3-4"
        if dl == 4 and lb == 2: return "Nickel (4-2-5)"
        if dl == 4 and lb == 1: return "Dime (4-1-6)"
        if dl == 0 and lb == 0: return "All-Out Blitz"
        return f"{dl}-{lb}"

    def _coverage_label(
        self, deep_ids: List[int], positions: Dict, los_y: float
    ) -> str:
        safeties = [positions[p] for p in deep_ids
                    if p in positions and (los_y - positions[p][1]) > SAFETY_DEPTH_MIN_PX]

        if not safeties:
            return "Cover-0 (Blitz)"
        if len(safeties) >= 2:
            spread = max(p[0] for p in safeties) - min(p[0] for p in safeties)
            if spread > 200:
                return "Cover-2 (Two-High)"
            return "Cover-4 (Quarters)"
        return "Cover-1 / Cover-3"

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _extract_positions(self, player_tracks: Dict) -> Dict:
        """Pull (x,y) from transformed position or bbox center."""
        out = {}
        for pid, data in player_tracks.items():
            pos = data.get("position_transformed")
            if pos:
                out[pid] = pos
                continue
            bbox = data.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                out[pid] = ((x1+x2)/2, (y1+y2)/2)
        return out

    def _filter_to_field_zone(self, positions: Dict) -> Dict:
        """Remove players outside the active field area (bench, sideline)."""
        return {
            pid: (x, y)
            for pid, (x, y) in positions.items()
            if FIELD_X1 <= x <= FIELD_X2 and FIELD_Y1 <= y <= FIELD_Y2
        }
