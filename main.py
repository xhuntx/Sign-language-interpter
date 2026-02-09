import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras

MODEL_PATH = "sign_numbers_classifier.h5"
model = keras.models.load_model(MODEL_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def extract_hand_landmarks(image_bgr, hands):
    """
    Run MediaPipe Hands on a BGR frame and return a 63-element feature vector
    [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20] for the first detected hand.

    This matches the feature format used in your dataset script.
    """
    # MediaPipe expects RGB input
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if not result.multi_hand_landmarks:
        return None, result

    # Use the first detected hand only
    hand_landmarks = result.multi_hand_landmarks[0]

    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])

    # coords length = 21 landmarks * 3 coords = 63
    return np.array(coords, dtype=np.float32), result


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # static_image_mode=False: video stream; tracking between frames
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            h, w, _ = frame_bgr.shape

            # 1) Extract features (63-dim) from the first hand, if any
            features, result = extract_hand_landmarks(frame_bgr, hands)

            prediction_text = "No hand detected"

            # 2) If we got landmarks, run your model
            if features is not None:
                # Model expects shape (batch_size, 63)
                input_batch = features[np.newaxis, :]
                preds = model.predict(input_batch, verbose=0)

                # preds shape: (1, num_classes) -> pick argmax
                class_idx = int(np.argmax(preds, axis=1)[0])

                # You trained labels as 0–9 for digits 1–10
                digit = class_idx + 1

                prediction_text = f"Predicted: {digit}"

            # 3) Draw landmarks for visualization (if any hands found)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_bgr,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # 4) Overlay prediction text
            cv2.putText(
                frame_bgr,
                prediction_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Sign Language Interpreter", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
