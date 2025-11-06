import os
import cv2
import numpy as np
import pickle
import tensorflow as tf

"""
General Helpers
"""
def load_model_and_labels(path="models/universal_emotion_model.keras"):
    """Load a trained Keras model and label mappings if available."""
    model = tf.keras.models.load_model(path)
    print("Model loaded:", path)

    # Load label mappings if present
    idx_to_emotion = None
    mapping_path = os.path.join("models", "universal_emotion_mappings.pkl")
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, "rb") as f:
                mappings = pickle.load(f)
            idx_to_emotion = mappings.get("idx_to_emotion")
            # Normalize keys to int if needed
            if idx_to_emotion and not isinstance(list(idx_to_emotion.keys())[0], int):
                idx_to_emotion = {int(k): v for k, v in idx_to_emotion.items()}
                print("Loaded label mappings")
        except Exception as e:
            print(f"Failed to load label mappings: {e}")

    return model, idx_to_emotion


def color_for_label(label: str) -> tuple:
    """Return a distinct BGR color for a given emotion label."""
    palette = {
        'angry': (0, 0, 255),         # Red
        'disgust': (0, 128, 0),       # Dark Green
        'fear': (128, 0, 128),        # Purple
        'happy': (0, 255, 255),       # Yellow (BGR)
        'sad': (255, 0, 0),           # Blue
        'surprise': (0, 165, 255),    # Orange
        'neutral': (200, 200, 200),   # Light Gray
        'silly': (255, 0, 255),       # Magenta
    }
    if label in palette:
        return palette[label]
    h = abs(hash(label)) % 255
    return (h, (h * 2) % 255, (h * 3) % 255)


def load_emotion_images():
    """Load per-emotion images from ./images directory into a dict."""
    base = os.path.join(os.path.dirname(__file__), "images")
    mapping = {}
    if not os.path.isdir(base):
        return mapping

    candidates = [
        'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'silly'
    ]
    try:
        for f in os.listdir(base):
            if f.lower().endswith('.png'):
                name = os.path.splitext(f)[0].lower()
                if name not in candidates:
                    candidates.append(name)
    except Exception:
        pass

    for name in candidates:
        path = os.path.join(base, f"{name}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                mapping[name] = img
    return mapping


def overlay_rgba(background, overlay, x, y):
    """Overlay an image onto background at (x, y)."""
    h, w = overlay.shape[:2]
    H, W = background.shape[:2]
    if x >= W or y >= H:
        return background
    w = min(w, W - x)
    h = min(h, H - y)
    if w <= 0 or h <= 0:
        return background
    overlay_roi = overlay[0:h, 0:w]
    bg_roi = background[y:y+h, x:x+w]
    if overlay_roi.shape[2] == 4:
        alpha = overlay_roi[:, :, 3] / 255.0
        alpha = np.dstack([alpha, alpha, alpha])
        fg = overlay_roi[:, :, :3].astype(float)
        bg = bg_roi.astype(float)
        out = cv2.convertScaleAbs(fg * alpha + bg * (1 - alpha))
        background[y:y+h, x:x+w] = out
    else:
        background[y:y+h, x:x+w] = overlay_roi
    return background
