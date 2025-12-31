import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

def create_kf(bbox):
    # bbox = [x1, y1, x2, y2]
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    w  = bbox[2] - bbox[0]
    h  = bbox[3] - bbox[1]

    kf = cv2.KalmanFilter(8, 4)

    # State transition
    kf.transitionMatrix = np.eye(8, dtype=np.float32)
    for i in range(4):
        kf.transitionMatrix[i, i+4] = 1.0

    # Measurement: cx, cy, w, h
    kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
    kf.measurementMatrix[0, 0] = 1
    kf.measurementMatrix[1, 1] = 1
    kf.measurementMatrix[2, 2] = 1
    kf.measurementMatrix[3, 3] = 1

    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(8, dtype=np.float32)

    kf.statePost = np.array(
        [[cx], [cy], [w], [h], [0], [0], [0], [0]],
        dtype=np.float32
    )

    return kf

class Track:
    def __init__(self, bbox, track_id):
        self.kf = create_kf(bbox)
        self.id = track_id
        self.time_since_update = 0
        self.hits = 1
        self.history = []   # center points
        self.speed = 0.0

def kf_to_bbox(state):
    cx, cy, w, h = state[:4].flatten()
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]

def center(bbox):
    return int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)

def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])

    return inter / (areaA + areaB - inter + 1e-6)


class ByteTracker:
    def __init__(self, high_th=0.6, low_th=0.1, iou_th=0.3):
        self.tracks = []
        self.next_id = 0
        self.high_th = high_th
        self.low_th = low_th
        self.iou_th = iou_th
    def predict(self):
        for t in self.tracks:
            t.kf.predict()
            t.time_since_update += 1
    def associate(self, detections, create_new):
        if not self.tracks or not detections:
            if create_new:
                for d in detections:
                    self.tracks.append(Track(d[:4], self.next_id))
                    self.next_id += 1
            return

        cost = np.zeros((len(self.tracks), len(detections)))
        for i, t in enumerate(self.tracks):
            tb = kf_to_bbox(t.kf.statePost)
            for j, d in enumerate(detections):
                cost[i, j] = 1 - iou(tb, d[:4])

        r, c = linear_sum_assignment(cost)

        matched_dets = set()
        for i, j in zip(r, c):
            if cost[i, j] < 1 - self.iou_th:
                self.update_track(self.tracks[i], detections[j])
                matched_dets.add(j)

        if create_new:
            for i, d in enumerate(detections):
                if i not in matched_dets:
                    self.tracks.append(Track(d[:4], self.next_id))
                    self.next_id += 1
    def update_track(self, track, det):
        x1,y1,x2,y2 = det[:4]
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        w = x2-x1
        h = y2-y1

        track.kf.correct(np.array([[cx],[cy],[w],[h]], dtype=np.float32))
        track.time_since_update = 0
        track.hits += 1

        bbox = kf_to_bbox(track.kf.statePost)
        c = center(bbox)
        track.history.append(c)

        if len(track.history) > 1:
            dx = track.history[-1][0] - track.history[-2][0]
            dy = track.history[-1][1] - track.history[-2][1]
            track.speed = (dx*dx + dy*dy) ** 0.5

    def cleanup(self, max_age=60):
        self.tracks = [t for t in self.tracks if t.time_since_update < max_age]

    def update(self, detections):
        high = [d for d in detections if d[4] >= self.high_th]
        low  = [d for d in detections if self.low_th <= d[4] < self.high_th]

        self.predict()
        self.associate(high, create_new=True)
        self.associate(low, create_new=False)
        self.cleanup()

        return self.tracks

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

model = YOLO("yolov8n.pt") # model for object detection
# we will use bytesort rather than deepsort cause i don't have gpu and its must faster!
tracker = ByteTracker()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    detections = []
    for r in results: 
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])
    tracks = tracker.update(detections)
    for t in tracks:
        if t.hits < 3:
            continue

        bbox = kf_to_bbox(t.kf.statePost)
        x1,y1,x2,y2 = map(int, bbox)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,
            f"ID:{t.id} Speed:{t.speed:.1f}",
            (x1,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        for i in range(1, len(t.history)):
            cv2.line(frame, t.history[i-1], t.history[i], (255,0,0), 2)
    LINE_Y = 200
    count = 0
    crossed = set()

    for t in tracks:
        if len(t.history) < 2:
            continue

        y_prev = t.history[-2][1]
        y_curr = t.history[-1][1]

        if y_prev < LINE_Y and y_curr >= LINE_Y and t.id not in crossed:
            crossed.add(t.id)
            count += 1

    cv2.line(frame, (0,LINE_Y),(frame.shape[1],LINE_Y),(0,0,255),2)
    cv2.putText(frame, f"Count:{count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("week 3 work", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    print(f"ID {t.id} | hits={t.hits} | history_len={len(t.history)}")

cap.release()
cv2.DestroyAllWindows()


