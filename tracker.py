import pickle
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from typing import List, Dict, Optional


CLASS_MAP = {
    0: "players",
    1: "referees",
    2: "ball",
}


class Tracker:
    def __init__(self, model_path: str = "yolov8x.pt"):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def get_object_tracks(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> Dict[str, List[Dict]]:

        if read_from_stub and stub_path:
            try:
                with open(stub_path, "rb") as f:
                    tracks = pickle.load(f)
                print(f"Loaded tracking stub from {stub_path}")
                return tracks
            except FileNotFoundError:
                print("Stub not found, running YOLO...")

        detections = self._detect_frames(frames)
        tracks = self._build_tracks(detections)

        if stub_path:
            import os
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
            print(f"Tracking cached → {stub_path}")

        return tracks

    def interpolate_ball_positions(self, ball_positions: List[Dict]) -> List[Dict]:
        import pandas as pd
        df = pd.DataFrame([
            {"frame": i,
             "x1": v.get(1, {}).get("bbox", [None]*4)[0],
             "y1": v.get(1, {}).get("bbox", [None]*4)[1],
             "x2": v.get(1, {}).get("bbox", [None]*4)[2],
             "y2": v.get(1, {}).get("bbox", [None]*4)[3]}
            for i, v in enumerate(ball_positions)
        ])
        df[["x1","y1","x2","y2"]] = df[["x1","y1","x2","y2"]].interpolate()
        df[["x1","y1","x2","y2"]] = df[["x1","y1","x2","y2"]].bfill()

        result = []
        for _, row in df.iterrows():
            if all(v is not None for v in [row.x1, row.y1, row.x2, row.y2]):
                result.append({1: {"bbox": [row.x1, row.y1, row.x2, row.y2]}})
            else:
                result.append({})
        return result

    def _detect_frames(self, frames: List[np.ndarray]):
        detections = []
        batch_size = 20
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            results = self.model.predict(batch, conf=0.2)
            for result in results:
                detections.append(sv.Detections.from_ultralytics(result))
        return detections

    def _build_tracks(self, detections) -> Dict[str, List[Dict]]:
        tracks = {"players": [], "referees": [], "ball": []}

        for detection in detections:
            # class 0 = person, class 32 = sports ball (COCO classes)
            player_det = detection[detection.class_id == 0]
            ball_det   = detection[detection.class_id == 32]

            player_det = self.tracker.update_with_detections(player_det)

            players_frame = {}
            for i in range(len(player_det)):
                track_id = player_det.tracker_id[i]
                bbox = player_det.xyxy[i]
                if track_id is not None:
                    players_frame[int(track_id)] = {"bbox": bbox.tolist()}

            ball_frame = {}
            if len(ball_det) > 0:
                ball_frame[1] = {"bbox": ball_det.xyxy[0].tolist()}

            tracks["players"].append(players_frame)
            tracks["referees"].append({})
            tracks["ball"].append(ball_frame)

        return tracks
