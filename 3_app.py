import pickle
import subprocess
import threading
import time
import cv2
import mediapipe as mp
import numpy as np

# Load trained model
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

labels_dict = {
    0: "Hello",
    1: "Yes",
    2: "No",
    3: "Victory",
    4: "Help"
}

is_speaking = False
last_spoken_sign = ""
last_spoken_time = 0

def speak_windows(text):
    global is_speaking
    try:
        cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
        subprocess.run(["powershell", "-Command", cmd], capture_output=True)
    except Exception:
        pass
    finally:
        is_speaking = False

def speak(text):
    global is_speaking
    if not is_speaking:
        is_speaking = True
        threading.Thread(target=speak_windows, args=(text,), daemon=True).start()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

current_candidate = ""
candidate_counter = 0

print("SignSpeak Engine Ready! Press 'Q' or 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    display_label = "Waiting for hand..."
    display_color = (0, 165, 255)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y

            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - wrist_x)
                data_aux.append(lm.y - wrist_y)

            probs = model.predict_proba([data_aux])[0]
            pred_idx = int(np.argmax(probs))
            conf = probs[pred_idx] * 100

            # Sirf 50% se upar confidence par lock karo
            if conf > 50:
                pred_sign = labels_dict.get(pred_idx, "Recognized")
                display_label = f"{pred_sign} ({conf:.0f}%)"
                display_color = (0, 255, 0)

                # Stabilizer: Hath kam se kam 8 frames tak lagataar wahi sign dikhaye tabhi lock hoga
                if pred_sign == current_candidate:
                    candidate_counter += 1
                else:
                    current_candidate = pred_sign
                    candidate_counter = 0

                # 8 frames confirm aur 1.5 second cooldown taaki repetitive/galat audio na trigger ho
                current_time = time.time()
                if candidate_counter >= 8:
                    if pred_sign != last_spoken_sign or (current_time - last_spoken_time > 2.0):
                        last_spoken_sign = pred_sign
                        last_spoken_time = current_time
                        speak(pred_sign)
            else:
                display_label = f"Scanning ({conf:.0f}%)"
                display_color = (0, 0, 255)
                candidate_counter = 0
    else:
        candidate_counter = 0
        current_candidate = ""

    # Top HUD banner
    cv2.rectangle(frame, (0, 0), (W, 70), (20, 20, 20), -1)
    cv2.putText(frame, f"Sign: {display_label}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, display_color, 2)

    cv2.imshow('SignSpeak - Live Detection', frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()