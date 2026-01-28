
# Day 1: Data Science - Basics of Classification 

For **Day 1**, I focused on understanding the **fundamentals of Classification** in Machine Learning. I go through all scikit-learn library to build a simple "Gender Predictor" based on physical attributes (Height, Weight, Shoe Size).

📋 Project Overview
My goal is to compare how different classification algorithms handle the same small dataset.

I manually created a dataset and trained 5 different models to see if they would agree on the prediction for a new user.

The Data
Features: [Height (cm), Weight (kg), Shoe Size]

Target: Gender ('male' or 'female')

🛠️ Tech Stack
Language: Python 
Library: Scikit-Learn (sklearn)

🧠 Models Implemented
I imported dependencies from various sklearn modules to test the following classifiers:

Decision Tree Classifier (tree) : it uses a flowchart like structure to make decisions 

K-Nearest Neighbors (neighbors) : based on the closed "k" data point it classifies.

Support Vector Machine (svm) : Finds the optimal boundary to seperate the classes.

Random Forest Classifier (ensemble) : Uses multiple DecisionTrees for Better acuuary .

Gaussian Naive Bayes (naive_bayes) : uses probability to predict .

💻 How to Run
Install Dependencies:

pip install scikit-learn
python main.py

Input Data: 
[Height (cm),Weight (kg),Shoe Size] 

📊 Example Output
Enter the height: 170
Enter the weight: 70
Enter the shoe size: 40

Decision Tree Classifier: ['female']
KNN Classifier: ['female']
SVM : ['female']
Random Forest Classifier: ['female']
Naive Bayes Classifier: ['female']

📝 Learning Outcomes
Learned how to import specific models from sklearn.
Understood the .fit() method for training and .predict() for prediction.
Observed that different models might gives different results depending on the data complexity.

----------------------------------------------------------------------------------------------------------------------------------------------------------
# Day 2: Linear Regression 

For **Day 2**, I moves from Classification to **Regression**. 
While in classification categorizes data (A vs B), but in Regression predicts continuous values (like prices, temperature, or scores).

📋 Project: Score Predictor
I built a simple Logic that predicts a student's test score based on the number of hours they studied.

🧠 The Math Behind Linear Regression
The model tries to fit a straight line through the data points using the equation:
$$y = mx + c$$
* **y**: The Score (Target)
* **x**: The Hours Studied (Feature)
* **m**: The Slope (How much the score goes up per hour)
* **c**: The Intercept (The score if you study 0 hours)

🛠️ Libraries Used
* **Scikit-Learn:** For the `LinearRegression` model.
* **NumPy:** To handle the array shapes.
* **Matplotlib:** To visualize the "Line of Best Fit."

📊 Visuals
The code generates a graph showing the actual student scores (Blue Dots) and the machine's learned pattern (Red Line).

----------------------------------------------------------------------------------------------------------------------------------------------------------
# Day 3: Logistic Regression - The "S" Curve 🧬

For **Day 3**, I learned **Logistic Regression**. 
Despite the name "Regression," this is actually a **Classification** algorithm used to predict binary outcomes (Yes/No, True/False, 0/1).

📋 Project: Insurance Predictor
I built a model that predicts whether a customer will buy insurance based on their age.

🧠 The Math: Sigmoid Function
Unlike Linear Regression which fits a straight line ($y=mx+c$), Logistic Regression applies the **Sigmoid Function** to squash the output between 0 and 1:
$$S(x) = \frac{1}{1 + e^{-x}}$$

* If the probability > 50%, it predicts **1 (Yes)**.
* If the probability < 50%, it predicts **0 (No)**.

🛠️ Libraries Used
* **Scikit-Learn:** `LogisticRegression`
* **Matplotlib:** To visualize the decision boundary (the S-Curve).

📊 Visuals
The red line in the graph represents the **Probability**.
* Bottom of the S: Low probability (Younger people).
* Top of the S: High probability (Older people).

----------------------------------------------------------------------------------------------------------------------------------------------------------
# Day 4: K-Nearest Neighbors (KNN)
For Day 4, I explored K-Nearest Neighbors (KNN). Unlike other models that "learn" a complex math formula, KNN is a "Lazy Learner"—it simply memorizes the entire dataset. When we ask for a prediction, it looks for the 'K' most similar data points (neighbors) and takes a vote.

📋 Project: Interactive T-Shirt Size Predictor
I built a CLI (Command Line Interface) tool that predicts whether a user needs a Small or Large T-shirt.

The program asks the user for their Height and Weight.

It compares the user to a dataset of 18 existing customers.

It predicts the size and plots the user as a Star on the graph.

🧠 Key Concept: Euclidean Distance
KNN calculates the straight-line distance between the new point and every other point to find the closest matches.
Small K (e.g., 1): Highly sensitive to noise.
Large K (e.g., 5): More stable, but might miss local details.
I used K=3 for this project.

🛠️ Libraries Used
Scikit-Learn: KNeighborsClassifier for the algorithm.
Matplotlib: To visualize the "Cluster" of Small vs. Large customers.
NumPy: For handling the data arrays.

📊 Visuals

🔵 Blue Dots: Small Size

🔴 Red Dots: Large Size

⭐ Yellow/Star: Entered value

📝 Learning Outcomes
Interactive Input: Learned to use input() and float() to make the model dynamic.
Visualization: Learned to plot a specific user marker (marker='*') on top of existing training data to visually confirm the model's decision.

----------------------------------------------------------------------------------------------------------------------------------------------------------
# Day 5: Random Forest 

For **Day 5**, I learned **Random Forest**, an "Ensemble" method.
Instead of relying on a single Decision Tree (which can be biased), Random Forest creates hundreds of trees and makes them vote on the final answer.

## 📋 Project: HR Hiring Predictor
I built an AI that mimics a Hiring Manager.
* **Input:** Experience, Test Score, Interview Score.
* **Output:** Hired vs. Rejected.

## 🧠 Key Concept: Feature Importance
One of the best features of Random Forest is that it is "interpretable." It can calculate exactly which input feature had the biggest impact on the decision. 
* *Example:* The model might reveal that "Interview Score" is 3x more important than "Years of Experience."

## 🛠️ Libraries Used
* **Scikit-Learn:** `RandomForestClassifier`
* **Matplotlib:** To visualize the "Importance Bar Chart."

----------------------------------------------------------------------------------------------------------------------------------------------------------
# Day 6: SVM(Support Vector Machine) Breast Cancer Prediction Project

This project is about predicting whether a breast tumor is dangerous (Malignant) or not dangerous (Benign) using a computer program.
The system looks at patient details like age, tumor size, and medical history and then gives a prediction.

🌟 Why This Project Is Important
Early detection of breast cancer can save lives
Helps doctors make better decisions
Reduces human error
Uses data to support medical diagnosis

📊 What Data Is Used
Dataset used: `https://www.kaggle.com/datasets/fatemehmehrparvar/breast-cancer-prediction`
The data contains information like:
Patient age
Year of diagnosis
Tumor size
Whether cancer has spread
Location of the tumor
Final diagnosis result
All personal details are removed.
The data is used only for learning and education.

🧹 What the Program Does (Step by Step)
Reads the patient data
Cleans incorrect and missing values
Converts text into numbers the computer can understand
Creates simple graphs to understand the data
Trains a machine learning model
Tests how accurate the prediction is
Shows results using charts and numbers

🤖 How the Prediction Works (In Simple Words)
The computer learns from past medical records
It finds patterns between patient details and diagnosis
Based on these patterns, it predicts the result for new data
Think of it like:
“Learning from previous cases to make a smart decision.”

📈 What Results You Can See
Accuracy of the prediction
A report showing how well the model performed
A confusion matrix chart
Graphs showing age and tumor size patterns
All graphs are saved automatically in the EDA_Images folder.

▶ How to Run This Project

1⃣ Install required software
Make sure Python is installed.
pip install pandas numpy matplotlib seaborn scikit-learn

2⃣ Run the program
python sample.py
That’s it 
The program will run and show results.

----------------------------------------------------------------------------------------------------------------------------------------------------------
# Day 7: K-Means Clustering - Unsupervised Learning 🛍️
For Day 7, I shifted from Supervised Learning  to Unsupervised Learning. I used the K-Means Clustering algorithm to find hidden patterns in raw data without any labels or guidance.

🧠 The Concept: "The Messy Room Party"
Imagine a party with 100 strangers. Without being told, they naturally separate into groups: the dancers near the speakers, the business people with wine, and the foodies near the buffet.
Supervised Learning: The host tells everyone where to stand.
Unsupervised Learning (K-Means): The guests group themselves based on similarity.

⚙️ The Architecture: How It Works
The algorithm uses an iterative process called Centroid Optimization:
Initialization: Drops K random "Centroids" (Flags) on the data map.
Assignment: Every data point connects to the closest Centroid.
Update: The Centroids move to the mathematical center (average) of their new team.
Repeat: Steps 2 & 3 repeat until the Centroids stop moving.

📋 Project: Mall Customer Segmentation
Dataset: Real customer data (Mall_Customers.csv). `https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python`
Features: Annual Income (k$) and Spending Score (1-100).
Goal: Identify specific "Customer Personas" to target with different marketing campaigns.

🛠️ The Pipeline
Data Loading: Ingesting the raw CSV.
The Elbow Method: Calculated WCSS (Within-Cluster Sum of Squares) for K=1 to K=10 to scientifically find the optimal number of groups.
Result: The "Elbow" (bend) occurred at K=5.
Model Training: Trained K-Means with 5 clusters.
Visualization: Plotted the 5 distinct market segments.
Prediction: Built a function to classify a new customer into one of these existing groups.

Interact: The script will first show the Elbow Graph, then the Cluster Map, and finally predict the group for a new customer.

📝 Learning Outcomes
Inertia (WCSS): Learned how to measure how "tight" a cluster is.
Centroids: Understood that a cluster is defined by its center point.
Business Logic: Translated mathematical clusters (0, 1, 2) into actionable business strategies.

----------------------------------------------------------------------------------------------------------------------------------------------------------
A huge thank you to **Siraj Raval** for the inspiration and guidance to start this journey.
* **GitHub:** [@llSourcell](https://github.com/llSourcell)
* **YouTube:** [Siraj Raval](https://youtu.be/T5pRlIbr6gg?si=O2xE08iO7yRP6U3c)

This project was built following his Data Science tutorials.
