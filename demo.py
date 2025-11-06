#!/usr/bin/env python3
import cv2
import time
import numpy as np
from utils import load_model_and_labels, load_emotion_images, color_for_label, overlay_rgba
from preprocess import preprocess_face

def main():
    # Load model and labels
    model, idx_to_emotion = load_model_and_labels()
    if not model:
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("Failed to load Haar cascade for face detection.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Real-time emotion detection running. Press 'Q' to quit.")
    emotion_images = load_emotion_images()

    last_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

            for (x, y, w, h) in faces:
                face_batch = preprocess_face(gray, (x, y, w, h))
                if face_batch is None:
                    continue

                try:
                    preds = model.predict(face_batch, verbose=0)
                    class_idx = int(np.argmax(preds))
                    conf = float(np.max(preds))
                    label = idx_to_emotion.get(class_idx, f"Class {class_idx}") if idx_to_emotion else f"Class {class_idx}"
                except Exception as e:
                    print(f"Prediction error: {e}")
                    continue

                # Draw bounding box and label
                color = color_for_label(label)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Overlay emotion icon
                key = label.lower() if label else None
                if key and key in emotion_images:
                    icon = emotion_images[key]
                    target_w = max(60, int(w * 0.5))
                    scale = target_w / icon.shape[1]
                    icon_resized = cv2.resize(icon, (target_w, int(icon.shape[0] * scale)), 
                                             interpolation=cv2.INTER_AREA)
                    frame = overlay_rgba(frame, icon_resized, x, max(0, y - icon_resized.shape[0] - 6))

            cv2.imshow("Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


