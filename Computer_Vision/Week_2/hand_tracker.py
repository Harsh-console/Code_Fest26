import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

start_time = time.monotonic()
frame_count = 0
fps = "not defined"
gesture = ""
while True:
    ret, frame = cap.read()
    frame_count +=1
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb) # contains 21 hands paarams
    if results.multi_hand_landmarks: 
        for hand_landmarks in results.multi_hand_landmarks: 
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            ) 
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        fingers_open = 0

        # Index
        if lm[8].y < lm[6].y:
            fingers_open += 1

        # Middle
        if lm[12].y < lm[10].y:
            fingers_open += 1

        # Ring
        if lm[16].y < lm[14].y:
            fingers_open += 1

        # Pinky
        if lm[20].y < lm[18].y:
            fingers_open += 1

        if fingers_open >= 4:
            gesture = "OPEN HAND"
        elif fingers_open <= 1:
            gesture = "FIST"
        else:
            gesture = "UNKNOWN"
    
    elapsed_time = time.monotonic() - start_time
    if elapsed_time >= 0.5:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.monotonic()

    text = f"{gesture} | FPS : {int(fps)}"

    cv2.putText(
    frame,
    text,
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
    )

    
    cv2.imshow("MediaPipe Hands", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



