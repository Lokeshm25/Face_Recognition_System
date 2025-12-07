#!/usr/bin/env python3
"""
Train SVM on FaceNet embeddings.

Usage (from repo root):
    python src/train_svm.py --dataset ./dataset/train --models_dir ./models
"""

import os
import argparse
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ensure local imports work when running `python src/train_svm.py`
import sys
sys.path.append(os.path.dirname(__file__))

from dataset_loader import FaceLoading
from embedder import FaceEmbedder


def train_svm(dataset_dir, models_dir):
    os.makedirs(models_dir, exist_ok=True)

    # 1) Load images
    loader = FaceLoading(dataset_dir)
    X_img, Y = loader.load_classes()
    print(f"\nTotal images loaded: {len(Y)}")

    # 2) Compute embeddings
    embedder = FaceEmbedder()
    EMBEDDED = [embedder.get_embedding(img) for img in X_img]
    EMBEDDED = np.asarray(EMBEDDED)

    # 3) Save embeddings (optional but useful)
    emb_path = os.path.join(models_dir, 'faces_embeddings.npz')
    np.savez_compressed(emb_path, EMBEDDED, Y)
    print(f"Saved embeddings at: {emb_path}")

    # 4) Encode labels
    encoder = LabelEncoder()
    Y_enc = encoder.fit_transform(Y)

    encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"Saved label encoder at: {encoder_path}")

    # 5) Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        EMBEDDED, Y_enc, shuffle=True, random_state=17
    )

    # 6) Train SVM
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, Y_train)

    # 7) Evaluate
    ypreds_train = model.predict(X_train)
    ypreds_test = model.predict(X_test)

    train_acc = accuracy_score(Y_train, ypreds_train)
    test_acc = accuracy_score(Y_test, ypreds_test)

    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")

    # 8) Save model
    model_path = os.path.join(models_dir, 'svm_model_160x160.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nSaved SVM model at: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='./dataset/train',
        help='Path to train dataset folder (each subfolder is a class/person).'
    )
    parser.add_argument(
        '--models_dir',
        type=str,
        default='./models',
        help='Where to save model files.'
    )
    args = parser.parse_args()

    train_svm(args.dataset, args.models_dir)


if __name__ == "__main__":
    main()
