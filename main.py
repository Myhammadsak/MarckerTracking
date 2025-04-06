import cv2
import numpy as np
from collections import OrderedDict

RED_LOWER = [np.array([0, 150, 100]), np.array([160, 150, 100])]
RED_UPPER = [np.array([10, 255, 255]), np.array([180, 255, 255])]

MIN_AREA = 40
MAX_AREA = 30000
MIN_RADIUS = 4
MAX_RADIUS = 150
MAX_MARKERS = 6


class StableMarkerTracker:
    def __init__(self):
        self.markers = OrderedDict()
        self.next_id = 0

    def _get_marker_key(self, x, y):
        return f"{int(x / 20)}:{int(y / 20)}"

    def update(self, detected_markers):
        for marker_id in self.markers:
            self.markers[marker_id]['active'] = False

        detected_markers_sorted = sorted(detected_markers, key=lambda m: m[1])

        for (x, y, r) in detected_markers_sorted:
            closest_id = None
            min_dist = float('inf')

            for marker_id, data in self.markers.items():
                dist = np.sqrt((x - data['x']) ** 2 + (y - data['y']) ** 2)
                if dist < 100 and dist < min_dist:
                    min_dist = dist
                    closest_id = marker_id

            if closest_id is not None:
                self.markers[closest_id].update({
                    'x': x, 'y': y, 'radius': r,
                    'active': True, 'missed': 0
                })
            elif len(self.markers) < MAX_MARKERS:
                marker_key = self._get_marker_key(x, y)
                self.markers[self.next_id] = {
                    'x': x, 'y': y, 'radius': r,
                    'key': marker_key, 'active': True, 'missed': 0
                }
                self.next_id += 1

        to_delete = []
        for marker_id, data in self.markers.items():
            if not data['active']:
                data['missed'] += 1
                if data['missed'] > 10:
                    to_delete.append(marker_id)

        for marker_id in to_delete:
            del self.markers[marker_id]


def enhance_red_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    red_mask1 = cv2.inRange(hsv, RED_LOWER[0], RED_UPPER[0])
    red_mask2 = cv2.inRange(hsv, RED_LOWER[1], RED_UPPER[1])
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    s = cv2.add(s, np.where(red_mask > 0, 50, 0).astype('uint8'))

    v = cv2.add(v, np.where(red_mask > 0, 30, 0).astype('uint8'))

    enhanced_hsv = cv2.merge([h, s, v])
    enhanced_frame = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    return enhanced_frame


def detect_markers(frame):
    enhanced_frame = enhance_red_color(frame)
    hsv = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, RED_LOWER[0], RED_UPPER[0])
    mask |= cv2.inRange(hsv, RED_LOWER[1], RED_UPPER[1])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2)

            if 0.7 < circularity < 1.3:
                if len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    (x, y), (ma, MA), angle = ellipse
                    radius = int((ma + MA) / 4)

                    if MIN_RADIUS <= radius <= MAX_RADIUS:
                        markers.append((int(x), int(y), radius))

    return markers


tracker = StableMarkerTracker()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    enhanced_frame = enhance_red_color(frame)
    cv2.imshow("Enhanced Red", enhanced_frame)

    current_markers = detect_markers(frame)
    tracker.update(current_markers)

    sorted_ids = sorted(tracker.markers.keys())
    for i in range(len(sorted_ids) - 1):
        if tracker.markers[sorted_ids[i]]['active'] and tracker.markers[sorted_ids[i + 1]]['active']:
            x1, y1 = tracker.markers[sorted_ids[i]]['x'], tracker.markers[sorted_ids[i]]['y']
            x2, y2 = tracker.markers[sorted_ids[i + 1]]['x'], tracker.markers[sorted_ids[i + 1]]['y']
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    for marker_id, data in tracker.markers.items():
        if data['active']:
            x, y, r = data['x'], data['y'], data['radius']
            color = (0, 255, 0) if data['missed'] == 0 else (0, 0, 255)
            cv2.circle(frame, (x, y), r, color, 2)
            cv2.putText(frame, f"ID:{marker_id}", (x - 20, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()