import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from .utils import crop_and_resize

class FaceLoading:
    def __init__(self, directory, target_size=(160,160)):
        self.directory = directory
        self.target_size = target_size
        self.detector = MTCNN()

    def extract_face_from_path(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(img)
        if not faces:
            raise ValueError("No face detected")
        box = faces[0]['box']
        face = crop_and_resize(img, box, self.target_size)
        return face

    def load_faces(self, subdir):
        FACES = []
        for im_name in os.listdir(subdir):
            path = os.path.join(subdir, im_name)
            try:
                single_face = self.extract_face_from_path(path)
                FACES.append(single_face)
            except Exception as e:
                # skip unreadable / no-face images
                continue
        return FACES

    def load_classes(self):
        X = []
        Y = []
        for sub_dir in sorted(os.listdir(self.directory)):
            full = os.path.join(self.directory, sub_dir)
            if not os.path.isdir(full):
                continue
            faces = self.load_faces(full)
            labels = [sub_dir] * len(faces)
            print(f"Loaded {len(labels)} images for class '{sub_dir}'")
            X.extend(faces)
            Y.extend(labels)
        return np.asarray(X), np.asarray(Y)


