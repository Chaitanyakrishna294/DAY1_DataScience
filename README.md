# DAY1_DataScience
Day 1: Data Science - Basics of Classification 🚀

For Day 1, I focused on understanding the fundamentals of Classification in Machine Learning. I go through all scikit-learn library to build a simple "Gender Predictor" based on physical attributes (Height, Weight, Shoe Size).

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

A huge thank you to **Siraj Raval** for the inspiration and guidance to start this journey.
* **GitHub:** [@llSourcell](https://github.com/llSourcell)
* **YouTube:** [Siraj Raval](https://youtu.be/T5pRlIbr6gg?si=O2xE08iO7yRP6U3c)

This project was built following his Data Science tutorials.
