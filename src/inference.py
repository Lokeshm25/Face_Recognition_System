#!/usr/bin/env python3
import argparse
import os
import cv2
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from embedder import FaceEmbedder
from utils import crop_and_resize


class FaceRecognizer:
    def __init__(self, model_path, encoder_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)
        self.detector = MTCNN()
        self.embedder = FaceEmbedder()

    def detect_faces(self, img_rgb):
        faces = self.detector.detect_faces(img_rgb)
        return faces

    def predict_single(self, img_path, show=True):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detect_faces(img_rgb)
        if not faces:
            print("No face found.")
            return None
        box = faces[0]['box']
        face = crop_and_resize(img_rgb, box)
        emb = self.embedder.get_embedding(face)
        probs = self.model.predict_proba([emb])[0]
        pred = self.model.predict([emb])[0]
        label = self.encoder.inverse_transform([pred])[0]
        if show:
            x,y,w,h = [int(v) for v in box]
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, f"{label} ({probs.max():.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            import matplotlib.pyplot as plt
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
        return label, probs.max()

    def find_in_group(self, img_path, target_name, show=True):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detect_faces(img_rgb)
        found = False
        for face in faces:
            box = face['box']
            face_crop = crop_and_resize(img_rgb, box)
            emb = self.embedder.get_embedding(face_crop)
            pred = self.model.predict([emb])[0]
            label = self.encoder.inverse_transform([pred])[0]
            if label == target_name:
                found = True
                x,y,w,h = [int(v) for v in box]
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(img, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        if show:
            import matplotlib.pyplot as plt
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
        return found

    def recognize_live(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detect_faces(img_rgb)
            for face in faces:
                box = face['box']
                face_crop = crop_and_resize(img_rgb, box)
                emb = self.embedder.get_embedding(face_crop)
                pred = self.model.predict([emb])[0]
                label = self.encoder.inverse_transform([pred])[0]
                x,y,w,h = [int(v) for v in box]
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.imshow("Live Face Recognition - press q to exit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single','group','live'], required=True)
    parser.add_argument('--image', type=str, help='path to image for single/group mode')
    parser.add_argument('--name', type=str, help='target name for group search')
    parser.add_argument('--model', type=str, default='models/svm_model_160x160.pkl')
    parser.add_argument('--encoder', type=str, default='models/label_encoder.pkl')
    args = parser.parse_args()

    recognizer = FaceRecognizer(args.model, args.encoder)
    if args.mode == 'single':
        if not args.image:
            raise ValueError("Provide --image for single mode")
        label, conf = recognizer.predict_single(args.image)
        print("Predicted:", label, "conf:", conf)
    elif args.mode == 'group':
        if not args.image or not args.name:
            raise ValueError("Provide --image and --name for group mode")
        found = recognizer.find_in_group(args.image, args.name)
        print(f"Found {args.name}: {found}")
    elif args.mode == 'live':
        recognizer.recognize_live()

if __name__ == "__main__":
    main()


