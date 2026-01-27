import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler

def savepic(name):
    plt.savefig(name,dpi=300,bbox_inches='tight')

df=pd.read_csv('breast-cancer-dataset.csv')
df.drop('S/N',axis=1,inplace=True)
df['Year']=pd.to_numeric(df["Year"],errors='coerce')
df['Tumor Size (cm)']=df['Tumor Size (cm)'].str.extract(r'(\d+)')
df['Tumor Size (cm)']=pd.to_numeric(df['Tumor Size (cm)'],errors='coerce')
df['Inv-Nodes']=pd.to_numeric(df['Inv-Nodes'],errors='coerce')
df['Metastasis']=pd.to_numeric(df['Metastasis'],errors='coerce')

nums_col=['Year','Tumor Size (cm)','Inv-Nodes','Metastasis']
for col in nums_col:
    df[col]=df[col].fillna(df[col].median())

df['Diagnosis Result']=df['Diagnosis Result'].map({'Benign':0,'Malignant':1})

sns.countplot(x='Diagnosis Result',data=df)
savepic('diagnosis_countplot.png')
plt.show()
sns.histplot(df['Age'],kde=True)
savepic('age_distribution.png')
plt.show()
sns.boxplot(x='Diagnosis Result',y='Tumor Size (cm)',data=df)
savepic('tumor_size_boxplot.png')
plt.show()


df=pd.get_dummies(df,drop_first=True)

X=df.drop('Diagnosis Result',axis=1)
y=df['Diagnosis Result']
scaler=StandardScaler()
X=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=SVC(kernel='rbf',C=1,gamma='scale',class_weight='balanced')
model.fit(X_train,y_train)

accuracy=model.score(X_test,y_test)
print(f"Model Accuracy: {accuracy*100:.2f}%")
y_pred=model.predict(X_test)
print("Accuracy Score:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred),sep="\n")
print("Classification Report:",classification_report(y_test,y_pred),sep="\n")     

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
savepic('confusion_matrix.png')
plt.show()

