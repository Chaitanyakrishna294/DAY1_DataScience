
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
A huge thank you to **Siraj Raval** for the inspiration and guidance to start this journey.
* **GitHub:** [@llSourcell](https://github.com/llSourcell)
* **YouTube:** [Siraj Raval](https://youtu.be/T5pRlIbr6gg?si=O2xE08iO7yRP6U3c)

This project was built following his Data Science tutorials.
