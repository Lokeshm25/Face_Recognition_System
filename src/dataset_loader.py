import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

# make sure we can import utils even when running `python src/train_svm.py`
import sys
sys.path.append(os.path.dirname(__file__))
from utils import crop_and_resize

class FaceLoading:
    """
    Loads faces from a dataset directory.

    Expected directory structure:
    dataset/train/
        person1/
            *.jpg
        person2/
            *.jpg
    """

    def __init__(self, directory, target_size=(160, 160)):
        self.directory = directory
        self.target_size = target_size
        self.detector = MTCNN()

    def extract_face_from_path(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(img_rgb)

        if not faces:
            raise ValueError("No face detected")

        box = faces[0]['box']
        face = crop_and_resize(img_rgb, box, self.target_size)
        return face

    def load_faces(self, subdir):
        faces = []
        for im_name in os.listdir(subdir):
            path = os.path.join(subdir, im_name)
            if not os.path.isfile(path):
                continue
            try:
                face = self.extract_face_from_path(path)
                faces.append(face)
            except Exception:
                # skip unreadable / no-face images
                continue
        return faces

    def load_classes(self):
        X = []
        Y = []

        # each subfolder = a class/label
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
