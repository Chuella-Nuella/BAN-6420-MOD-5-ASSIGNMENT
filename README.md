README: 
Assignment 5 - PCA and Logistic Regression on Breast Cancer Dataset
Course: BAN6420 (Programming in Python and R)

Project Description
This project demonstrates the use of Principal Component Analysis (PCA) to identify key features in the Breast Cancer dataset and applies logistic regression for classification. The goal is to assist Anderson Cancer Center in analyzing breast cancer cases and classifying them as either malignant (cancerous) or benign (non-cancerous) based on digital images of breast tissue.

Tasks Overview
1. Task 0: PCA Implementation
Apply PCA to extract essential variables from the dataset provided by sklearn.datasets.
2. Task 1: Dimensionality Reduction
Reduce the dataset to two principal components using PCA.
3. Task 2: Bonus Task (Optional)
Use logistic regression to predict whether a case is malignant or benign.

Key Steps
1. Data Preparation
Imported the Breast Cancer dataset from sklearn.datasets.
Standardized the features using StandardScaler to ensure equal contribution from all variables.
2. PCA Implementation
Decomposed the dataset into two principal components.
Identified key features contributing to each principal component.
Examined the explained variance ratio and cumulative variance to assess the effectiveness of dimensionality reduction.
3. Logistic Regression
Implemented logistic regression on the reduced dataset.
Trained the model on the training set and evaluated its performance on the test set.
4. Performance Evaluation
Achieved an accuracy score of 97.08%, indicating the model’s excellent performance in classifying breast cancer cases.
Visualized the PCA components and cumulative explained variance.
Usage Instructions
1. Clone or Download the Repository
Download the project files or clone the repository:
git clone <repository-link>
2. Install Dependencies
Install the required Python libraries:
pip install -r requirements.txt
3. Run the Script
Execute the main script:
python main.py
4. View Results
PCA scatter plot and cumulative variance plot will be displayed.
Model accuracy and classification report will be printed in the console.

Key Findings
1. Dimensionality Reduction:
PCA successfully reduced the dataset to two principal components while retaining most of the variance.
2. Model Performance:
The logistic regression model achieved an impressive accuracy of 97.08%, classifying breast cancer cases effectively.
3. Feature Importance:
PCA identified key variables contributing to each principal component, providing insights into the dataset.

Implications and Future Work

Implications
This project highlights the potential of machine learning techniques like PCA and logistic regression in healthcare, particularly for accurate breast cancer diagnosis.

Future Work
1. Enhancing Model Performance:
Explore advanced algorithms like support vector machines (SVM) or ensemble methods for improved accuracy.
2. Integration with Medical Imaging:
Combine this model with imaging technologies for real-time cancer detection.
3. Clinical Validation:
Validate the model's effectiveness in clinical settings through trials.
Project Structure
Assignment5/
│
├── main.py                  # Main Python script
├── requirements.txt         # Required Python libraries
├── README.md                # Project documentation
├── plots/
│   ├── pca_scatterplot.png  # Scatter plot of PCA components
│   ├── cumulative_variance.png # Cumulative variance plot
└── results/
    ├── classification_report.txt # Model classification report

Acknowledgements
The dataset used in this project is part of the Breast Cancer dataset available through sklearn.datasets.
# BAN-6420-MOD-5-ASSIGNMENT
