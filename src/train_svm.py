#!/usr/bin/env python3
"""
Train SVM on FaceNet embeddings (run locally or in Colab).
Produces: models/svm_model_160x160.pkl and models/label_encoder.pkl
"""
import os
import argparse
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from dataset_loader import FaceLoading
from embedder import FaceEmbedder

def main(args):
    dataset_dir = args.dataset
    models_dir = args.models_dir
    os.makedirs(models_dir, exist_ok=True)

    loader = FaceLoading(dataset_dir)
    X_img, Y = loader.load_classes()
    print(f"Total images loaded: {len(Y)}")
    embedder = FaceEmbedder()
    EMBEDDED = [embedder.get_embedding(img) for img in X_img]
    EMBEDDED = np.asarray(EMBEDDED)

    # Save embeddings optionally
    np.savez_compressed(os.path.join(models_dir, 'faces_embeddings.npz'), EMBEDDED, Y)

    # Encode labels
    encoder = LabelEncoder()
    Y_enc = encoder.fit_transform(Y)
    with open(os.path.join(models_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(encoder, f)
    print("Saved label encoder.")

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED, Y_enc, shuffle=True, random_state=17)

    # SVM
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, Y_train)
    ypreds_train = model.predict(X_train)
    ypreds_test = model.predict(X_test)

    print("Train acc:", accuracy_score(Y_train, ypreds_train))
    print("Test acc:", accuracy_score(Y_test, ypreds_test))

    with open(os.path.join(models_dir, 'svm_model_160x160.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print("Saved model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='path to train dataset folder (each subfolder = class)')
    parser.add_argument('--models_dir', type=str, default='models', help='where to save model files')
    args = parser.parse_args()
    main(args)


