import sys, os, warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
# warnings.filterwarnings("ignore")        # Suppress Python warnings
# sys.stderr = open(os.devnull, 'w')       # Suppress MediaPipe warnings

from keras.layers import TFSMLayer
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the Teachable Machine model
model = TFSMLayer("model1", call_endpoint="serving_default")

# Load class names
with open("model1/labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20

            # Keep within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:
                resized = cv2.resize(hand_img, (224, 224))
                img_array = np.expand_dims(resized / 255.0, axis=0)

                predictions = model(img_array)
                output_tensor = predictions['sequential_3'].numpy()
                class_index = np.argmax(output_tensor)
                confidence = output_tensor[0][class_index]

                if confidence >= 0.6:
                    class_name = class_names[class_index]
                    cv2.putText(frame, f"{class_name} ({confidence*100:.1f}%)",
                                (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show Exit instruction
    cv2.putText(frame, "Press 'Esc' to exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Hand Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
