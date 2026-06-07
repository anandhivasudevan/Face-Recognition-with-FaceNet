#  Face Recognition with FaceNet

> A deep learning pipeline for accurate face recognition using FaceNet embeddings, MTCNN face detection, and a Linear SVM classifier — implemented in a Jupyter Notebook.

---

##  Overview

This project implements a complete **face recognition system** using the **FaceNet** model developed by Google. FaceNet maps face images into a compact 128-dimensional Euclidean space where distances directly reflect face similarity. By combining FaceNet embeddings with a Support Vector Machine (SVM) classifier, the system can efficiently identify and verify individuals across different images.

The entire pipeline is built inside a Jupyter Notebook, making it easy to explore, visualize, and extend.

---

##  Features

-  **MTCNN Face Detection** — Accurate multi-stage face localization and alignment
-  **FaceNet Embeddings** — 128-dimensional face representations via a pre-trained deep CNN
-  **SVM Classification** — Linear SVM trained on embeddings to identify individuals
- **Model Evaluation** — Accuracy reporting on a validation set
-  **Notebook-based Workflow** — Interactive, step-by-step pipeline for easy experimentation

---

##  Project Structure

```
Face-Recognition-with-FaceNet/
├── face-recognition-with-facenet (1).ipynb   # Main notebook — full pipeline
└── README.md
```

---

## Tech Stack

| Component         | Tool / Library                        |
|-------------------|---------------------------------------|
| Language          | Python 3                              |
| Environment       | Jupyter Notebook                      |
| Face Detection    | MTCNN (Multi-task Cascaded CNNs)      |
| Face Embedding    | FaceNet (pre-trained deep CNN)        |
| Classifier        | Linear SVM (`scikit-learn`)           |
| Deep Learning     | TensorFlow / Keras                    |
| Data Processing   | NumPy, OpenCV / PIL, Matplotlib       |

---

##  Pipeline Workflow

### 1.  Face Detection
Uses **MTCNN** to detect and localize faces in input images. Only the cropped face regions are passed forward, improving accuracy and reducing noise.

### 2. Face Embedding Generation
The detected faces are fed into the pre-trained **FaceNet** model, which outputs a **128-dimensional embedding vector** per face. These compact vectors encode the unique identity of each face.

### 3.  Data Preparation
Embeddings are split into **training** and **validation** datasets to facilitate supervised classifier training and unbiased evaluation.

### 4.  Classifier Training
A **Linear SVM** from `scikit-learn` is trained on the face embeddings. The high-dimensional feature space provided by FaceNet makes SVM a highly effective choice for separating identities.

### 5.  Model Evaluation
The trained SVM is evaluated on the validation set. Accuracy metrics confirm the effectiveness of the FaceNet + SVM approach for face recognition.

---

##  Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/anandhivasudevan/Face-Recognition-with-FaceNet.git
cd Face-Recognition-with-FaceNet

# Install dependencies
pip install tensorflow keras mtcnn scikit-learn numpy opencv-python pillow matplotlib
```

### Running the Notebook

```bash
jupyter notebook "face-recognition-with-facenet (1).ipynb"
```

Then run all cells in order to execute the full pipeline.

---

##  How FaceNet Works

FaceNet is trained using a **triplet loss function**:

```
Loss = max(0, ||f(anchor) - f(positive)||² - ||f(anchor) - f(negative)||² + margin)
```

- **Anchor** — a reference face image
- **Positive** — a different image of the same person
- **Negative** — an image of a different person

The model learns to minimize intra-class distances and maximize inter-class distances, producing a highly discriminative embedding space where faces of the same person cluster tightly together.

---

##  Performance

The system reports classification accuracy on the validation set. Combining FaceNet's powerful 128-D embeddings with a Linear SVM yields strong recognition performance even with limited training data per identity.

---

##  References

- [FaceNet: A Unified Embedding for Face Recognition and Clustering (arXiv)](https://arxiv.org/abs/1503.03832)
- [MTCNN Face Detection](https://github.com/ipazc/mtcnn)
- [How to Develop a Face Recognition System Using FaceNet in Keras — Machine Learning Mastery](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/)
- [FaceNet — Wikipedia](https://en.wikipedia.org/wiki/FaceNet)

---

##  License

This project is open source. See the [LICENSE](LICENSE) file for details.



