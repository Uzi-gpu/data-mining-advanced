# 📊 Data Mining & Advanced Topics

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Advanced AI/ML topics including **Data Mining**, **Neural Network Theory**, and **Comprehensive Image Processing** demonstrating deep understanding of machine learning fundamentals and advanced techniques.

---

## 📋 Table of Contents
- [Projects Overview](#-projects-overview)
- [Technologies Used](#️-technologies-used)
- [Installation](#-installation)
- [Project Details](#-project-details)
- [Key Concepts](#-key-concepts)
- [Contact](#-contact)

---

## 🚀 Projects Overview

| # | Project | Category | Notebook | Focus |
|---|---------|----------|----------|-------|
| 1 | **Data Mining Project** | Data Science | [`01_data_mining_project.ipynb`](01_data_mining_project.ipynb) | Clustering, Association Rules |
| 2 | **ANN Loss Functions** | Deep Learning Theory | [`02_ann_loss_functions.ipynb`](02_ann_loss_functions.ipynb) | Softmax, Sigmoid, Cross-Entropy |
| 3 | **Image Processing** | Computer Vision | [`03_comprehensive_image_processing.ipynb`](03_comprehensive_image_processing.ipynb) | Complete CV Pipeline |
| 4 | **ML Comprehensive Exam** | Machine Learning | [`04_ml_comprehensive_exam.ipynb`](04_ml_comprehensive_exam.ipynb) | End-to-End ML Tasks |

---

## 🛠️ Technologies Used

### Data Mining
- **scikit-learn** - Clustering, classification
- **Pandas** - Data manipulation
- **Association rule mining** - Market basket analysis

### Deep Learning
- **TensorFlow/Keras** - Neural networks
- **Loss functions** - Optimization theory
- **Activation functions** - Softmax, Sigmoid, ReLU

### Image Processing
- **OpenCV** - Computer vision
- **PIL/Pillow** - Image manipulation
- **NumPy** - Array operations

---

## 📦 Installation

```bash
git clone https://github.com/uzi-gpu/data-mining-advanced.git
cd data-mining-advanced
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
jupyter notebook
```

---

## 📊 Project Details

### 1. 📈 Data Mining Project

**File:** [`01_data_mining_project.ipynb`](01_data_mining_project.ipynb)

**Objective:** Apply data mining techniques to discover patterns in data

**Techniques:**
- **Clustering**: K-Means, Hierarchical
- **Classification**: Decision Trees, Random Forest
- **Association Rules**: Apriori algorithm
- **Pattern Discovery**: Frequent itemsets

**Applications:**
- Customer segmentation
- Market basket analysis
- Anomaly detection
- Recommendation systems

---

### 2. 🧠 ANN Loss Functions

**File:** [`02_ann_loss_functions.ipynb`](02_ann_loss_functions.ipynb)

**Objective:** Deep dive into neural network loss functions and optimization

**Loss Functions Covered:**

**1. Binary Cross-Entropy:**
```python
BCE = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
```
- Use case: Binary classification
- Range: [0, ∞)

**2. Categorical Cross-Entropy:**
```python
CCE = -Σ(y_i * log(ŷ_i))
```
- Use case: Multi-class classification
- Requires: One-hot encoded labels

**3. Mean Squared Error (MSE):**
```python
MSE = (1/n) * Σ(y - ŷ)²
```
- Use case: Regression
- Sensitive to outliers

**Activation Functions:**
- **Sigmoid**: σ(x) = 1/(1+e^(-x))
- **Softmax**: e^(x_i) / Σe^(x_j)
- **ReLU**: max(0, x)

---

### 3. 🖼️ Comprehensive Image Processing

**File:** [`03_comprehensive_image_processing.ipynb`](03_comprehensive_image_processing.ipynb)

**Objective:** Complete image processing pipeline from basics to advanced

**Topics Covered:**

**Fundamentals:**
- Image loading and display
- Color space conversions
- Image resizing and cropping

**Filtering:**
- Gaussian blur
- Median filtering
- Bilateral filter
- Sharpening

**Edge Detection:**
- Canny edge detector
- Sobel operator
- Laplacian

**Morphological Operations:**
- Erosion and dilation
- Opening and closing
- Morphological gradient

**Advanced:**
- Histogram equalization
- Image transforms (FFT)
- Feature detection (corners, blobs)
- Image segmentation

---

### 4. 🎯 ML Comprehensive Exam

**File:** [`04_ml_comprehensive_exam.ipynb`](04_ml_comprehensive_exam.ipynb)

**Objective:** Demonstrate comprehensive ML knowledge

**Skills Demonstrated:**
- Data preprocessing
- Model selection
- Hyperparameter tuning
- Cross-validation
- Performance evaluation
- Feature engineering

---

## 📚 Key Concepts Demonstrated

### Data Mining
1. **Unsupervised Learning** - Clustering without labels
2. **Association Rules** - Mining relationships
3. **Pattern Discovery** - Finding hidden insights
4. **Dimensionality Reduction** - PCA, t-SNE

### Deep Learning Theory
1. **Loss Functions** - Optimization objectives
2. **Backpropagation** - Gradient computation
3. **Activation Functions** - Non-linearity
4. **Optimization** - SGD, Adam, RMSprop

### Image Processing
1. **Spatial Domain** - Direct pixel manipulation
2. **Frequency Domain** - FFT transformations
3. **Feature Extraction** - Corners, edges, textures
4. **Image Enhancement** - Filters, equalization

### Machine Learning
1. **Model Evaluation** - Accuracy, precision, recall
2. **Cross-Validation** - K-fold validation
3. **Ensemble Methods** - Bagging, boosting
4. **Feature Selection** - Important variable identification

---

## 🎓 Learning Outcomes

This repository demonstrates:

1. **Data Mining Expertise**
   - Clustering algorithms
   - Association rule mining
   - Pattern discovery
   - Practical applications

2. **Deep Learning Theory**
   - Loss function mathematics
   - Optimization principles
   - Activation function analysis
   - Training dynamics

3. **Image Processing**
   - Complete CV pipeline
   - Filter design and application
   - Feature extraction
   - Advanced techniques

4. **ML Proficiency**
   - End-to-end pipelines
   - Model evaluation
   - Best practices
   - Production readiness

---

## 📧 Contact

**Uzair Mubasher** - BSAI Graduate

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/uzair-mubasher-208ba5164)
[![Email](https://img.shields.io/badge/Email-uzairmubasher5@gmail.com-red)](mailto:uzairmubasher5@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-uzi--gpu-black)](https://github.com/uzi-gpu)

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

**⭐ Star this repository if you found it helpful!**
