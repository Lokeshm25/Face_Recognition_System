# Face Recognition System

A **Face Recognition** project using:

- **MTCNN** for face detection  
- **FaceNet** (via `keras-facenet`) for generating 512-D face embeddings  
- **SVM** classifier for identity recognition  

It supports:

- Identifying a person from a **single image**
- Searching for a specific person in a **group photo**
- **Live webcam** face recognition

---

## Project Structure

```text
Face_Recognition_System/
├── dataset/
│   ├── train/
│   │   ├── jenna_ortega/
│   │   ├── lokesh_maheshwari/
│   │   ├── robert_downey/
│   │   ├── sardor_abdirayimov/
│   │   └── taylor_swift/
│   └── test/
│       ├── test1.jpg
│       ├── ...
│       └── test11.jpg
├── models/
│   ├── label_encoder.pkl
│   └── svm_model_160x160.pkl
├── src/
│   ├── dataset_loader.py
│   ├── embedder.py
│   ├── inference.py
│   ├── train_svm.py
│   └── utils.py
├── requirements.txt
├── README.md
└── .gitignore

dataset/train/<person_name>/ — images for each person (used for training).

dataset/test/*.jpg — images for quick manual testing.

models/ — contains the trained SVM model and label encoder.

src/ — all Python source files.


Installation

It’s recommended to use a virtual environment.

cd Face_Recognition_System

python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt


Make sure models/ exists (you already have it):

mkdir -p models

Training

You already have models/label_encoder.pkl and models/svm_model_160x160.pkl.
Training is optional now, but you can re-train anytime if you change/add data.

To train (or re-train) the SVM with the current dataset:

python src/train_svm.py --dataset ./dataset/train --models_dir ./models


This will:

Load faces from dataset/train/<person_name>/

Extract FaceNet embeddings

Train an SVM classifier

Save:

models/faces_embeddings.npz

models/label_encoder.pkl

models/svm_model_160x160.pkl

You’ll see train and test accuracy printed in the terminal.

Inference / Usage

All commands below assume you are in the project root:
Face_Recognition_System/

1. Single Image Recognition

Identify the person in a single image:

python src/inference.py --mode single --image ./dataset/test/test1.jpg


Detects the face in test1.jpg

Outputs the predicted label + confidence

Displays the image with a bounding box and predicted name

2. Search a Person in a Group Photo

Check if a particular person (by name) appears in a group photo:

python src/inference.py \
  --mode group \
  --image ./dataset/test/test2.jpg \
  --name "lokesh_maheshwari"


--name must exactly match one of the folder names in dataset/train/

e.g. "jenna_ortega", "taylor_swift", "robert_downey", "sardor_abdirayimov", "lokesh_maheshwari"

If found, the script:

Draws a bounding box with the predicted label

Prints Found lokesh_maheshwari: True

3. Live Webcam Face Recognition

Use your webcam to recognize faces in real-time:

python src/inference.py --mode live


Opens a window showing a live video stream

Detected faces are annotated with names + confidence

Press q to close the window and stop

If your default camera is not working, you can change the camera index inside inference.py:

cap = cv2.VideoCapture(1)  # or 2, etc.

Requirements

From requirements.txt (ensure these are included):

opencv-python
numpy
matplotlib
mtcnn
keras-facenet
scikit-learn


You can add extra dev tools (like jupyter, ipykernel) if you want to experiment in notebooks.

Notes & Gotchas

Run from project root:
Always run commands from Face_Recognition_System/ so relative paths (./dataset, ./models) work correctly.

Face detection failures:
Some images (side profiles, noisy backgrounds, tiny faces) may not yield any detections with MTCNN. Those will either be skipped in training or return “No face found” in inference.

First run of FaceNet / MTCNN:
On the first run, keras-facenet and MTCNN may download model weights. Ensure internet access at least once.

Labels:
Class labels come directly from folder names in dataset/train/.
Renaming a folder changes the label text used during prediction.

Possible Improvements / Future Work

Add a confidence threshold to ignore low-confidence predictions (e.g. if conf < 0.6, show “Unknown”).

Add a Gradio web demo where users can upload an image and see predicted labels.

Expose a FastAPI endpoint for integration with other apps (send image → get JSON result).

Log accuracy & confusion matrix for more detailed evaluation.

Implement incremental update: add new person without retraining from scratch (by reusing embeddings).

License

This project is licensed under the MIT License.
See the LICENSE file for details.