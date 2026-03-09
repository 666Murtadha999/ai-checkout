import math


class SimpleTracker:
    """
    Very simple centroid-based tracker.
    It assigns a track_id to each detection and tries to keep the same id
    across frames based on distance.
    """

    def __init__(self, max_distance: float = 60.0):
        self.max_distance = max_distance
        self.next_id = 1
        self.tracks = {}  # track_id -> (cx, cy)

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def update(self, detections):
        """
        detections: list of dicts with key "bbox"
        returns: same list but each dict has "track_id"
        """
        new_tracks = {}
        used_track_ids = set()

        for det in detections:
            bbox = det.get("bbox")
            if bbox is None:
                continue

            cx, cy = self._centroid(bbox)

            # find nearest existing track
            best_id = None
            best_dist = None

            for track_id, (tx, ty) in self.tracks.items():
                if track_id in used_track_ids:
                    continue
                dist = math.hypot(cx - tx, cy - ty)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_id = track_id

            # if close enough, reuse id
            if best_id is not None and best_dist is not None and best_dist <= self.max_distance:
                track_id = best_id
            else:
                track_id = self.next_id
                self.next_id += 1

            used_track_ids.add(track_id)
            new_tracks[track_id] = (cx, cy)
            det["track_id"] = track_id

        self.tracks = new_tracks
        return detections
