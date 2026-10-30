import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# 5 solid, visually distinct gestures (Viva ke liye perfect aur 100% accurate)
labels_map = {
    0: "Hello",
    1: "Yes",
    2: "No",
    3: "Victory",
    4: "Help"
}
number_of_classes = len(labels_map)
dataset_size = 80

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

all_data = []
all_labels = []

print("=== STARTING DATA COLLECTION ===")

for j in range(number_of_classes):
    print(f"\nPrepare for gesture: [{labels_map[j]}]. Press 'Q' when your hand is ready in frame!")

    # Ready wait loop
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, f"Gesture: {labels_map[j]}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, "Get in position and Press 'Q' to Record", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('SignSpeak - Data Collector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Wrist ko origin (0,0) maan kar relative normalize karte hain
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                data_aux = []
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - wrist_x)
                    data_aux.append(lm.y - wrist_y)

                all_data.append(data_aux)
                all_labels.append(j)
                counter += 1

        cv2.putText(frame, f"Recording {labels_map[j]}: {counter}/{dataset_size}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow('SignSpeak - Data Collector', frame)
        cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': all_data, 'labels': all_labels}, f)

print("\nAll data recorded and saved into data.pickle!")
