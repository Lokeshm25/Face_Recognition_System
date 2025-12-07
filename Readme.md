# Face Recognition System

**Short:** Face recognition using MTCNN for detection + FaceNet embeddings + SVM classifier.  
Works in three modes: identify from single image, search a person in a group photo, or live webcam recognition.

---

## Features
- Detects faces with **MTCNN**.
- Embeds faces using **FaceNet** (`keras-facenet`).
- Lightweight SVM classifier (fast, good for small datasets).
- Modes: `single image`, `group search`, `live webcam`.
- Minimal dependencies and CLI-friendly scripts for reproducible results.

---

## Repo structure
face-recognition-system/
├── src/ # core scripts
├── models/ # trained models (store here or load from Drive)
├── data/dataset/ # train dataset (each subfolder is a person)
├── requirements.txt
└── README.md