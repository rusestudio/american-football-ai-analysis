"""
tactician.py
=============
Uses Claude API (Vision + Text) to analyze individual plays
and generate an actionable coaching report.

Claude receives:
  - Pre-snap frame image (base64) from drone footage
  - Detected formation labels (offense + defense)
  - Yards gained result
  - Player speed data
  - Sequence of plays so far (for pattern recognition)

Claude returns:
  - Play-by-play tactical commentary
  - Pattern alerts (e.g. "3rd-and-short: opponent runs QB sneak 80% of time")
  - Strategic recommendations for the coaching staff
  - A full game summary report in Markdown
"""

import os
import json
import base64
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import anthropic
import cv2
import numpy as np


class ClaudeTactician:
    """
    AI tactical analyst powered by Claude.

    Requires ANTHROPIC_API_KEY environment variable.
    """

    MODEL = "claude-sonnet-4-20250514"

    def __init__(self):
        self.client       = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.play_history: List[Dict] = []         # running context across plays

    # ── Per-Play Analysis ─────────────────────────────────────────────────────

    def analyze_play(self, play_data: Dict[str, Any]) -> str:
        """
        Analyze a single play with Claude Vision.

        Sends the pre-snap frame + structured data.
        Returns a coaching-focused commentary string.
        """
        self.play_history.append({
            "play_number":       play_data["play_number"],
            "offense_formation": play_data["offense_formation"],
            "defense_formation": play_data["defense_formation"],
            "yards_gained":      play_data.get("yards_gained"),
        })

        # Encode pre-snap frame as base64 JPEG
        image_b64 = self._frame_to_base64(play_data["pre_snap_frame"])

        # Build context from recent plays
        recent_context = self._build_play_history_context()

        prompt = f"""You are an experienced American football coach and tactical analyst.
You are reviewing drone top-down footage of a game.

## Current Play — Play #{play_data['play_number']}

**Detected formations (pre-snap):**
- Offense: {play_data['offense_formation']}
- Defense: {play_data['defense_formation']}

**Result:** {self._yards_label(play_data.get('yards_gained'))}

**Top player speeds this play (yards/second):**
{self._format_speeds(play_data.get('player_speeds', {}))}

## Recent Play History
{recent_context}

## Your Task
Looking at the top-down pre-snap image and the data above, provide:

1. **Formation Read** (2 sentences): What is each side trying to do with this alignment?
2. **Matchup Edge** (1-2 sentences): Which team had the structural advantage pre-snap?
3. **Result Analysis** (1-2 sentences): Was the result consistent with the formation matchup?
4. **Coaching Tip** (1 sentence): One specific adjustment the offense OR defense should make next time they face this alignment.

Keep it concise and actionable. Use football terminology the coaching staff will understand."""

        try:
            message = self.client.messages.create(
                model=self.MODEL,
                max_tokens=600,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type":   "image",
                            "source": {
                                "type":       "base64",
                                "media_type": "image/jpeg",
                                "data":       image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return message.content[0].text

        except Exception as e:
            return f"[Claude analysis unavailable: {e}]"

    # ── Pattern Recognition ───────────────────────────────────────────────────

    def detect_tendencies(self) -> str:
        """
        After multiple plays, ask Claude to identify opponent tendencies.
        Call this mid-game (e.g. after every 5 plays) for real-time coaching.
        """
        if len(self.play_history) < 3:
            return "Not enough plays yet to identify tendencies."

        history_json = json.dumps(self.play_history, indent=2)

        prompt = f"""You are an NFL-level defensive coordinator reviewing opponent tendencies.

Here is the play-by-play data from the game so far:
{history_json}

Analyze this sequence and identify:

1. **Run/Pass Tendency by Formation**: Which offensive formations correlate with runs vs passes?
2. **Down-and-Distance Patterns**: Any formation tendencies on 3rd-and-short, 3rd-and-long?
3. **Red Zone Tendencies**: (If any red zone plays detected)
4. **Defensive Vulnerability**: Which defensive formations gave up the most yards?
5. **Recommended Adjustments**: 2-3 specific in-game adjustments the coaching staff should make NOW.

Be direct and specific. Coaching staff needs to act on this immediately."""

        try:
            message = self.client.messages.create(
                model=self.MODEL,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            return f"[Tendency analysis unavailable: {e}]"

    # ── Full Game Report ──────────────────────────────────────────────────────

    def generate_full_report(
        self,
        play_analyses: List[Dict[str, Any]],
        output_dir: str = "reports/",
    ) -> str:
        """
        Generate a complete coaching report in Markdown.
        Saved to output_dir with timestamp.

        Returns the file path.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Aggregate stats
        total_plays   = len(play_analyses)
        yards_list    = [p["yards_gained"] for p in play_analyses
                         if p.get("yards_gained") is not None]
        avg_yards     = round(sum(yards_list) / len(yards_list), 1) if yards_list else 0
        total_yards   = round(sum(yards_list), 1) if yards_list else 0

        formations_used = {}
        for p in play_analyses:
            key = p["offense_formation"]
            formations_used[key] = formations_used.get(key, 0) + 1

        top_formation = max(formations_used, key=formations_used.get) if formations_used else "N/A"

        # Ask Claude for executive summary
        exec_summary = self._generate_executive_summary(
            play_analyses, avg_yards, total_yards, top_formation
        )

        # Tendencies analysis
        tendencies = self.detect_tendencies()

        # Build Markdown report
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        report_md  = self._build_report_markdown(
            timestamp, play_analyses, exec_summary,
            tendencies, total_yards, avg_yards,
            total_plays, formations_used
        )

        # Save
        report_path = Path(output_dir) / f"coaching_report_{timestamp}.md"
        report_path.write_text(report_md, encoding="utf-8")

        return str(report_path)

    def _generate_executive_summary(
        self,
        play_analyses, avg_yards, total_yards, top_formation
    ) -> str:
        history_json = json.dumps([
            {
                "play":      p["play_number"],
                "offense":   p["offense_formation"],
                "defense":   p["defense_formation"],
                "yards":     p.get("yards_gained"),
            }
            for p in play_analyses
        ], indent=2)

        prompt = f"""You are writing the executive summary for a post-game coaching report.

Game data:
- Total plays analyzed: {len(play_analyses)}
- Total yards: {total_yards}
- Average yards per play: {avg_yards}
- Most used offensive formation: {top_formation}

Play-by-play log:
{history_json}

Write a 3-paragraph executive summary:
1. Overall offensive performance assessment
2. Defensive performance and vulnerabilities exposed
3. Top 3 priority adjustments for next game

Write in the style of a professional NFL coordinator's report. Be specific."""

        try:
            message = self.client.messages.create(
                model=self.MODEL,
                max_tokens=700,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            return f"[Executive summary unavailable: {e}]"

    def _build_report_markdown(
        self, timestamp, play_analyses, exec_summary,
        tendencies, total_yards, avg_yards, total_plays, formations_used
    ) -> str:
        lines = [
            f"# 🏈 Coaching Analysis Report",
            f"**Generated:** {timestamp}  ",
            f"**Powered by:** Claude AI + Drone Vision Analysis",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            exec_summary,
            "",
            "---",
            "",
            "## Game Statistics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Plays Analyzed | {total_plays} |",
            f"| Total Yards | {total_yards} |",
            f"| Avg Yards/Play | {avg_yards} |",
            f"| Most Used Formation | {max(formations_used, key=formations_used.get) if formations_used else 'N/A'} |",
            "",
            "### Formation Frequency",
            "",
        ]

        for formation, count in sorted(formations_used.items(),
                                        key=lambda x: -x[1]):
            pct = round(count / total_plays * 100)
            lines.append(f"- **{formation}**: {count} plays ({pct}%)")

        lines += [
            "",
            "---",
            "",
            "## Tendency & Pattern Analysis",
            "",
            tendencies,
            "",
            "---",
            "",
            "## Play-by-Play Breakdown",
            "",
        ]

        for p in play_analyses:
            yards_label = self._yards_label(p.get("yards_gained"))
            lines += [
                f"### Play {p['play_number']}",
                f"**Offense:** {p['offense_formation']}  ",
                f"**Defense:** {p['defense_formation']}  ",
                f"**Result:** {yards_label}  ",
                "",
                p.get("claude_analysis", "_No analysis available_"),
                "",
                "---",
                "",
            ]

        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert OpenCV frame to base64 JPEG string."""
        # Resize to reduce API payload (720p is plenty for formation reading)
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)))

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")

    def _build_play_history_context(self) -> str:
        if len(self.play_history) <= 1:
            return "This is the first play."
        recent = self.play_history[-5:]  # last 5 plays for context
        lines  = []
        for p in recent[:-1]:  # exclude current play
            lines.append(
                f"  Play {p['play_number']}: {p['offense_formation']} vs "
                f"{p['defense_formation']} → {self._yards_label(p['yards_gained'])}"
            )
        return "\n".join(lines)

    @staticmethod
    def _yards_label(yards) -> str:
        if yards is None:
            return "Unknown"
        if yards > 0:
            return f"+{yards} yards gain"
        if yards < 0:
            return f"{yards} yards loss"
        return "No gain"

    @staticmethod
    def _format_speeds(player_speeds: Dict) -> str:
        if not player_speeds:
            return "  No speed data available."
        sorted_speeds = sorted(player_speeds.items(), key=lambda x: -x[1])[:5]
        return "\n".join(
            f"  Player #{pid}: {speed:.1f} yds/s"
            for pid, speed in sorted_speeds
        )
