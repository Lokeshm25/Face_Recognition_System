#!/usr/bin/env python3
"""
Inference script for the face recognition system.

Modes:
    single  - predict identity from a single face image
    group   - search for a given name inside a group photo
    live    - live webcam recognition

Usage examples (from repo root):
    python src/inference.py --mode single --image ./dataset/test/test1.jpg
    python src/inference.py --mode group --image ./dataset/test/test2.jpg --name "lokesh_maheshwari"
    python src/inference.py --mode live
"""

import os
import argparse
import pickle
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

# ensure local imports work when running `python src/inference.py`
import sys
sys.path.append(os.path.dirname(__file__))

from embedder import FaceEmbedder
from utils import crop_and_resize


class FaceRecognizer:
    def __init__(self, model_path, encoder_path):
        # load SVM model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # load LabelEncoder
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)

        self.detector = MTCNN()
        self.embedder = FaceEmbedder()

    def _predict_face_embedding(self, face_img):
        emb = self.embedder.get_embedding(face_img)
        probs = self.model.predict_proba([emb])[0]
        pred_idx = self.model.predict([emb])[0]
        label = self.encoder.inverse_transform([pred_idx])[0]
        return label, probs.max()

    def predict_single(self, img_path, show=True):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(img_rgb)

        if not faces:
            print("No face found in image.")
            return None, None

        box = faces[0]['box']
        face = crop_and_resize(img_rgb, box)

        label, conf = self._predict_face_embedding(face)

        if show:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{label} ({conf:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
            import matplotlib.pyplot as plt
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        return label, conf

    def find_in_group(self, img_path, target_name, show=True):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(img_rgb)

        if not faces:
            print("No faces found in image.")
            return False

        found = False
        for face in faces:
            box = face['box']
            face_crop = crop_and_resize(img_rgb, box)
            label, conf = self._predict_face_embedding(face_crop)

            if label == target_name:
                found = True
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{label} ({conf:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

        if show:
            import matplotlib.pyplot as plt
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        return found

    def recognize_live(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam.")
            return

        print("Live face recognition started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_faces(img_rgb)

            for face in faces:
                box = face['box']
                face_crop = crop_and_resize(img_rgb, box)
                label, conf = self._predict_face_embedding(face_crop)

                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} ({conf:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Live Face Recognition - press 'q' to exit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=['single', 'group', 'live'],
        required=True,
        help='Inference mode.'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image for single/group mode.'
    )
    parser.add_argument(
        '--name',
        type=str,
        help='Target name for group mode (must match train folder name).'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='./models/svm_model_160x160.pkl',
        help='Path to trained SVM model.'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='./models/label_encoder.pkl',
        help='Path to label encoder pickle.'
    )
    args = parser.parse_args()

    recognizer = FaceRecognizer(args.model, args.encoder)

    if args.mode == 'single':
        if not args.image:
            raise ValueError("You must provide --image for single mode.")
        label, conf = recognizer.predict_single(args.image)
        if label is not None:
            print(f"Predicted: {label} (confidence: {conf:.2f})")

    elif args.mode == 'group':
        if not args.image or not args.name:
            raise ValueError("You must provide --image and --name for group mode.")
        found = recognizer.find_in_group(args.image, args.name)
        print(f"Found {args.name}: {found}")

    elif args.mode == 'live':
        recognizer.recognize_live()


if __name__ == "__main__":
    main()
