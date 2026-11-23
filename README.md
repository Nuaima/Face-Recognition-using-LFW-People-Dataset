# Face Classification with the LFW People Dataset

A notebook-based image-classification exercise using the Labeled Faces in the Wild (LFW) dataset and a fully connected neural network built with TensorFlow/Keras.

## Overview

The notebook loads face images through scikit-learn's `fetch_lfw_people`, explores the class distribution, selects a subset of examples, normalizes the image vectors, and trains a dense neural network to classify images among the represented identities.

Although the repository name uses “face recognition,” the implementation is a supervised identity-classification experiment. It does **not** implement face detection, face embeddings, similarity matching, or a production identity-verification system.

## Workflow

1. Load LFW images with a minimum of 100 images per person.
2. Inspect image dimensions, identity labels, and class distribution.
3. Select up to 100 examples per identity for the experiment.
4. Normalize image pixel values and one-hot encode labels.
5. Create a stratified train/test split.
6. Train a fully connected Keras classifier.
7. Plot training/validation accuracy and inspect a confusion matrix.

## Technology

- Python
- scikit-learn (LFW dataset and data splitting)
- TensorFlow / Keras
- NumPy and Pandas
- Matplotlib and Seaborn

## Repository Contents

```text
.
├── Facial_Recognition_CNN.ipynb
├── Workflow.pdf
└── README.md
```

## Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Nuaima/Face-Recognition-using-LFW-People-Dataset.git
   cd Face-Recognition-using-LFW-People-Dataset
   ```
2. Install compatible versions of Python, scikit-learn, TensorFlow, NumPy, Pandas, Matplotlib, and Seaborn.
3. Open `Facial_Recognition_CNN.ipynb` in Jupyter or Google Colab.
4. Run the notebook from top to bottom. scikit-learn downloads the LFW dataset when it is not already cached.

## Evaluation

The notebook visualizes training and validation accuracy and generates a confusion matrix for the held-out split. No single accuracy figure is promoted here because the repository does not document a reproducible benchmark protocol or independent evaluation.

## Limitations and Responsible Use

- This is a small educational classification experiment, not a secure face-recognition or authentication solution.
- The notebook uses a limited subset of LFW examples and low-resolution grayscale images.
- Results can be affected by class imbalance, identity overlap in the source dataset, image conditions, and demographic coverage.
- Face data is sensitive biometric information. Do not use this project to identify, track, authenticate, or make consequential decisions about people.
