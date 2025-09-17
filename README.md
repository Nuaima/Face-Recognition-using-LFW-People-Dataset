# Face Recognition using LFW People Dataset

### Overview
This project demonstrates face recognition using the Labeled Faces in the Wild (LFW) dataset. The goal is to classify faces of people who appear at least 100 times in the dataset using a fully connected neural network.

---

### Technologies
- **Language:** Python  
- **Libraries:**  
  - Data Handling: `numpy`, `pandas`  
  - Visualization: `matplotlib`, `seaborn`  
  - Machine Learning / Deep Learning: `tensorflow`, `keras`  
  - Preprocessing & Utilities: `sklearn`  
- **Techniques:**  
  - Exploratory Data Analysis (EDA) with bar plots of class distribution  
  - Image normalization  
  - One-hot encoding for categorical labels  
  - Neural network classification using `Sequential` Keras models  
  - Model evaluation using training/validation accuracy plots and confusion matrices  

---

### Dataset
- **Source:** `fetch_lfw_people` from `sklearn.datasets`  
- **Description:** Faces of individuals appearing at least 100 times in the dataset.  
- **Number of Classes:** Variable depending on `min_faces_per_person`  
- **Image Dimensions:** 62 x 47 pixels  

---

### Notebook
- `Facial_Recognition_CNN.ipynb`: Contains full workflow including data loading, preprocessing, model building, training, evaluation, and visualization.

---

