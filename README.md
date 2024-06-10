# 📊 Advanced Performance Comparison of Classifiers

This project involves implementing, training, and evaluating multiple machine learning models before and after applying Principal Component Analysis (PCA). The goal is to determine the impact of PCA on model performance across various classifiers.

---

## 🎯 Project Overview

### Goal
Compare the performance of different classifiers before and after PCA to understand its impact on model accuracy.

---

## ⚙️ Steps and Implementation

### 1. Data Preparation
- **Libraries Used:**
  - `pandas` for data manipulation and analysis
  - `numpy` for numerical operations
  - `scikit-learn` for machine learning models and preprocessing
  - `matplotlib.pyplot` and `seaborn` for data visualization

### 2. Apply PCA
- Implement PCA to reduce the dimensionality of the dataset.
- Compare classifiers' performance before and after PCA.

### 3. Train and Evaluate Classifiers
- **Classifiers Implemented:**
  - k-Nearest Neighbors (k-NN)
  - Logistic Regression (LR)
  - Gaussian Naive Bayes (GNB)
  - Support Vector Machine (SVM)
  - Decision Tree (DT)
  - Random Forest (RF)
  - Gradient Boosting Trees (GBT)
  - Multi-Layer Perceptron (MLP)

---

## 📊 Results

### Performance Comparison
#### Graph:  

![image](https://github.com/MarawanEldeib/Advanced_Classifier_Performance_Analysis/assets/105850133/fb19e1fd-0255-4bbd-be68-6b0374490f44)


### Classifier Performance Before and After PCA
| Classifier            | Before PCA Accuracy | After PCA Accuracy |
|-----------------------|---------------------|--------------------|
| k-NN                  | 0.76                | 0.75               |
| Logistic Regression   | 0.95                | 0.92               |
| Gaussian Naive Bayes  | 0.65                | 0.70               |
| Support Vector Machine| 0.93                | 0.91               |
| Decision Tree         | 0.78                | 0.76               |
| Random Forest         | 0.89                | 0.87               |
| Gradient Boosting     | 0.88                | 0.85               |
| MLP Classifier        | 0.91                | 0.89               |

### Detailed Performance Metrics
| Metric                | Value               |
|-----------------------|---------------------|
| Optimal Clusters      | 6                   |
| Mean Accuracy         | 0.965               |
| Accuracy              | 0.995               |
| F1 Score              | 0.995               |
| Precision             | 0.995               |
| Recall                | 0.995               |
| Mean Squared Error    | 0.005               |
| R-Squared             | 0.998               |

**Confusion Matrix:**

| 23 | 0 | 0 | 0 | 0 | 0 |   
| 0  | 45 | 0 | 0 | 0 | 0 |   
| 0  | 0 | 33 | 0 | 0 | 0 |   
| 0  | 0 | 0 | 38 | 1 | 0 |   
| 0  | 0 | 0 | 0 | 39 | 0 |   
| 0  | 0 | 0 | 0 | 0 | 21 |   

### Discussion
The results indicate that PCA generally decreases the accuracy of most classifiers slightly. However, some classifiers like Gaussian Naive Bayes showed improvement. The impact of PCA varies across different classifiers, emphasizing the importance of model-specific preprocessing techniques.

### Conclusion
This study highlights the significance of PCA in dimensionality reduction and its varying impact on different classifiers. Understanding these effects can guide the selection of preprocessing techniques for different machine learning models.

---

## 🚀 Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/MarawanEldeib/Advanced_Performance_Comparison_of_Classifiers.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Advanced_Performance_Comparison_of_Classifiers
    ```

3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook Assignment_2_Classifier_Comparison.ipynb
    ```

4. Download the dataset:
    ```bash
    wget [classifier_performance.csv]
    ```

---

## 📦 Libraries
To replicate this project, ensure you have the following libraries installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```


![image](https://github.com/MarawanEldeib/Advanced_Classifier_Performance_Analysis/assets/105850133/bd4cfded-330e-4239-99c7-ed8c9b5e1c6c)

![image](https://github.com/MarawanEldeib/Advanced_Classifier_Performance_Analysis/assets/105850133/b0f508cf-10bc-498c-af5c-aa4a3674eb1e)

