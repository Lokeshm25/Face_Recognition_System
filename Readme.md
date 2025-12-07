
# Face Recognition System

A lightweight and fast **Face Recognition** system built using:

- **MTCNN** — for face detection  
- **FaceNet (keras-facenet)** — for generating 512-D embeddings  
- **SVM Classifier** — for identity prediction  

This project supports:

- 🔍 **Single Image Recognition**  
- 👥 **Person Search in Group Photos**  
- 🎥 **Real-time Webcam Recognition**

---

## 📁 Project Structure

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
└── README.md
````

### Folder Purposes

* **dataset/train/<person_name>/** — training images per identity
* **dataset/test/** — images for testing/inference
* **models/** — trained SVM model + label encoder
* **src/** — all Python source code

---

## 🚀 Installation

It is recommended to use a Python virtual environment.

```bash
cd Face_Recognition_System

python -m venv venv

# Linux / macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure the models directory exists:

```bash
mkdir -p models
```

---

## 🧠 Training the Model

Your repo already includes:

* `models/label_encoder.pkl`
* `models/svm_model_160x160.pkl`

So **training is optional** unless you modify the dataset.

### Train (or Re-train)

```bash
python src/train_svm.py --dataset ./dataset/train --models_dir ./models
```

This will:

1. Load face images from `dataset/train/<person_name>/`
2. Detect + crop faces using MTCNN
3. Generate FaceNet embeddings
4. Train the SVM classifier
5. Save:

   * `models/faces_embeddings.npz`
   * `models/label_encoder.pkl`
   * `models/svm_model_160x160.pkl`

Train + test accuracy will be printed.

---

## 🔎 Inference / Usage

All commands assume you are inside the project root:

```
Face_Recognition_System/
```

---

### **1️⃣ Single Image Recognition**

```bash
python src/inference.py --mode single --image ./dataset/test/test1.jpg
```

✔ Detects the face
✔ Prints predicted label + confidence
✔ Displays the image with bounding box & name

---

### **2️⃣ Search for a Person in a Group Photo**

```bash
python src/inference.py \
  --mode group \
  --image ./dataset/test/test2.jpg \
  --name "lokesh_maheshwari"
```

**Important:** `--name` **must match exactly** one of the folder names in `dataset/train/`.

Example valid names:

* `jenna_ortega`
* `taylor_swift`
* `robert_downey`
* `sardor_abdirayimov`
* `lokesh_maheshwari`

If the person is found:

* Bounding box is drawn
* Terminal prints:

  ```
  Found lokesh_maheshwari: True
  ```

---

### **3️⃣ Real-Time Webcam Face Recognition**

```bash
python src/inference.py --mode live
```

* Opens webcam
* Annotates detected faces with labels + confidence
* Press **q** to exit

If needed, change camera index in `inference.py`:

```python
cap = cv2.VideoCapture(1)
```

---

## 📦 Requirements

From **requirements.txt**:

```
opencv-python
numpy
matplotlib
mtcnn
keras-facenet
scikit-learn
```

Optional (for development):

```
jupyter
ipykernel
```

---

## ⚠️ Notes & Gotchas

### ✔ Always run from repo root

Relative paths like `./dataset` and `./models` will break otherwise.

### ✔ Face Detection May Fail

MTCNN may not detect:

* side profiles
* tiny faces
* heavy occlusions
* blurry / low-light images

Such images are skipped during training or return “No face found” during inference.

### ✔ First Run Downloads Weights

`keras-facenet` and MTCNN download pretrained weights on first use. Ensure internet access at least once.

### ✔ Labels Come From Folder Names

Changing a folder name changes the predicted label.

Example:

```
dataset/train/taylor_swift/  → label used: "taylor_swift"
```

---

## 💡 Future Improvements

* Add **confidence thresholding** (e.g., if conf < 0.60 → “Unknown”)
* Add a **Gradio UI** for quick demos (upload image → get label)
* Build a **FastAPI endpoint** (image upload → JSON result)
* Add evaluation metrics (confusion matrix, precision, recall)
* Implement **incremental updates** to add identities without full retraining

---

## 📄 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

```
```
