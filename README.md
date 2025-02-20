---

# **User Profiling in Social Media Using Text, Images, and Relational Data**

This project leverages machine learning techniques to predict user attributes—**gender, age group, and Big Five personality traits**—from Facebook data. It integrates multiple data sources, including textual status updates, profile images, and user-page interactions, to build robust predictive models.

---

## **Table of Contents**
- [Overview](#overview)  
- [Dataset](#dataset)  
- [Methodology](#methodology)  
  - [Text-Based Models](#text-based-models)  
  - [Image-Based Models](#image-based-models)  
  - [Relational Data Models](#relational-data-models)  
- [Results](#results)  
- [Technologies Used](#technologies-used)  
- [Future Work](#future-work)  
- [How to Run](#how-to-run)  

---

## **Overview**
This project explores multi-source data to predict user characteristics:
- **Gender prediction**: Binary classification using text, images, and relational data.
- **Age group prediction**: Classification into predefined age intervals.
- **Personality trait prediction**: Regression for Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism.

The project utilizes **machine learning and deep learning models**, including SVMs, CNNs, Random Forest, and Multinomial Naive Bayes, achieving high accuracy and interpretability.

---

## **Dataset**
The dataset comprises **9,500 labeled instances**, collected through the MyPersonality program. It includes:  
- **Text Data**: User status updates  
- **Image Data**: Profile pictures  
- **Relational Data**: User-page interactions (likes)  

Each instance includes labels for gender, age group, and Big Five personality traits.

---

## **Methodology**

### **Text-Based Models**
- **Support Vector Machine (SVM)** achieved **80% accuracy** and **75% macro F1-score** for gender prediction, outperforming transformer-based models.  
- **Personality Trait Prediction**: Ensemble models (Random Forest + XGBoost) reduced RMSE by **10%** compared to standalone models.  

### **Image-Based Models**
- **Convolutional Neural Networks (CNN)** for gender classification improved accuracy from **63% to 70%** with optimized architecture and standardized pre-processing.  

### **Relational Data Models**
- **Multinomial Naive Bayes** achieved **80% accuracy and 78% macro precision** for gender prediction, outperforming Random Forest (70%) and CART (70%).  

---

## **Results**
- **Gender Prediction (Text)**: 80% accuracy, 75% macro F1  
- **Gender Prediction (Images)**: Improved accuracy to 70%  
- **Gender Prediction (Relational Data)**: 80% accuracy, 78% macro precision  
- **Personality Trait Prediction**: 10% reduction in RMSE through ensemble methods  

---

## **Technologies Used**
- **Programming Languages**: Python  
- **Libraries and Frameworks**:  
  - Scikit-learn  
  - TensorFlow & Keras  
  - OpenCV  
  - XGBoost  
- **Data Processing**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn  

---

## **Future Work**
- Incorporate transformer-based models like DistilBERT for improved text-based personality prediction.  
- Explore deeper CNN architectures for image-based gender classification.  
- Expand relational data analysis to predict additional user attributes.  

---

## **How to Run**
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/username/user-profiling-social-media.git
   cd user-profiling-social-media
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Code**  
   - For Text-Based Models:  
     ```bash
     python text_model.py
     ```
   - For Image-Based Models:  
     ```bash
     python image_model.py
     ```
   - For Relational Data Models:  
     ```bash
     python relation_model.py
     ```

---

## **Contributors**
- **Rohan Avireddy** – Machine Learning Models and Text Analysis  
- **Carla Peterson** – Image Processing and CNN Models  
- **Blaise Dmello** – Relational Data Models and Integration  

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for more details.

---
