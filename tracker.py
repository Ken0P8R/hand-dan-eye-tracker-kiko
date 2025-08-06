import cv2
import mediapipe as mp
import numpy as np

# Init MediaPipe modules
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# TURUNKAN RESOLUSI AGAR LEBIH RINGAN
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155,
                133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249,
                 263, 466, 388, 387, 386, 385, 384, 398]

def draw_text_with_outline(img, text, pos, font, scale, color, thickness, outline_color, outline_thickness):
    cv2.putText(img, text, pos, font, scale, outline_color, thickness + outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)

def is_finger_up(hand_landmarks, tip, pip, handedness):
    if tip == mp_hands.HandLandmark.THUMB_TIP:
        wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
        thumb_tip_x = hand_landmarks.landmark[tip].x
        return thumb_tip_x < wrist_x if handedness == "Right" else thumb_tip_x > wrist_x
    return hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y

def is_only_middle_finger_up(hand_landmarks, handedness):
    tips = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP]
    pips = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.PINKY_PIP]
    status = [is_finger_up(hand_landmarks, tip, pip, handedness) for tip, pip in zip(tips, pips)]
    return status[2] and all(not s for i, s in enumerate(status) if i != 2)

if not cap.isOpened():
    print("Kamera tidak berhasil dibuka")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_hands = hands.process(frame_rgb)
    results_pose = pose.process(frame_rgb)
    results_face = face_mesh.process(frame_rgb)

    status_tangan = {"Right": "Tidak terdeteksi", "Left": "Tidak terdeteksi"}
    jari_tengah_only_terdeteksi = False

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            pts_left = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE_IDX]
            pts_right = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE_IDX]
            cv2.polylines(frame, [np.array(pts_left, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=1)
            cv2.polylines(frame, [np.array(pts_right, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=1)

    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            handedness = results_hands.multi_handedness[idx].classification[0].label

            if is_only_middle_finger_up(hand_landmarks, handedness):
                jari_tengah_only_terdeteksi = True

            fingers_status = [is_finger_up(hand_landmarks, tip, pip, handedness) for tip, pip in zip(
                [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                 mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP],
                [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                 mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP])]

            status = "Terbuka" if sum(fingers_status) >= 4 else "Tertutup"
            status_tangan[handedness] = status

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1)
            )

            wrist = hand_landmarks.landmark[0]
            for landmark_id in [1, 5, 9, 13, 17]:
                joint = hand_landmarks.landmark[landmark_id]
                x1, y1 = int(wrist.x * w), int(wrist.y * h)
                x2, y2 = int(joint.x * w), int(joint.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark

        for side in ["left", "right"]:
            if side == "left":
                wrist_id, elbow_id, shoulder_id = 15, 13, 11
            else:
                wrist_id, elbow_id, shoulder_id = 16, 14, 12

            wrist = landmarks[wrist_id]
            elbow = landmarks[elbow_id]
            shoulder = landmarks[shoulder_id]

            if wrist.visibility > 0.5 and elbow.visibility > 0.5:
                x1, y1 = int(wrist.x * w), int(wrist.y * h)
                x2, y2 = int(elbow.x * w), int(elbow.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if elbow.visibility > 0.5 and shoulder.visibility > 0.5:
                x1, y1 = int(elbow.x * w), int(elbow.y * h)
                x2, y2 = int(shoulder.x * w), int(shoulder.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    draw_text_with_outline(frame, f"Tangan Kanan: {status_tangan['Right']}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, (0, 0, 0), 2)
    draw_text_with_outline(frame, f"Tangan Kiri: {status_tangan['Left']}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, (0, 0, 0), 2)

    cv2.imshow("Hand, Arm & Eye Tracker", frame)

    if jari_tengah_only_terdeteksi:
        print("pakyu coding.")
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
