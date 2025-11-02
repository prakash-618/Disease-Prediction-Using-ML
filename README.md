# Disease Prediction Using Machine Learning

This project predicts diseases based on user symptoms using multiple machine learning models — **Support Vector Machine (SVM)**, **Naive Bayes**, and **Random Forest**.  
The models are trained, evaluated, and combined using majority voting for robust predictions.

---

## Steps

### 1. Data Preprocessing
- Encode columns (e.g., Disease)
- Fill missing values
- Resample dataset to handle class imbalance
- Flatten target variable if needed

### 2. Model Training and Evaluation
- Models used: Decision Tree, Random Forest, SVM, Naive Bayes  
- Evaluated using **Stratified K-Fold Cross-Validation**
- Metrics: Accuracy and Confusion Matrix

### 3. Ensemble Model
- Combines predictions from all three models using the **mode** (majority voting)
- Improves robustness and stability

### 4. Final Prediction Function
Takes comma-separated symptoms as input and predicts the most likely disease.

```python
print(predict_disease("skin_rash,fever,headache"))
