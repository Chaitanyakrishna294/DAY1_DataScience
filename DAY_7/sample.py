import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans

def saveimg(name):
    plt.savefig(name,dpi=300)
    plt.show()

df=pd.read_csv('Mall_Customers.csv')
df['Gender']=df['Gender'].map({'Male':0,'Female':1})
X=df.iloc[:,[3,4]].values
print(X.shape)
#WCSS = Within-Cluster Sum of Squares
#elbow method to find the k optimal no of clusters
wcss=[]
for i in range(1,11):
    model=KMeans(n_clusters=i,init='k-means++',random_state=42)
    model.fit(X)
    wcss.append(model.inertia_)
print(wcss)

plt.figure(figsize=(10,5))
plt.plot(range(1,11),wcss,marker='o',color='red')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS') 
plt.grid(True)
saveimg('WCSS.png')

model=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_kmeans=model.fit_predict(X)

plt.figure(figsize=(12, 7))

colors = ['red', 'blue', 'green', 'cyan', 'magenta']

for i in range(5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], 
                s=100, c=colors[i], label=f'Cluster {i}')
    
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], 
            s=300, c='yellow', marker='*', edgecolor='black', label='Centroids')

plt.title('Mall Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
saveimg('Customer_Segments.png')
try:
    A_income=int(input("Enter Annual Income (k$): "))
    S_score=int(input("Enter Spending Score (1-100): "))
except ValueError:
    print("invalid Input enter the Correct Values.")
    exit()
new_data=np.array([[A_income,S_score]])
predict=model.predict(new_data)
print(f"New Customer belongs to clustor {predict[0]}")

for i in range(5):
    plt.scatter(X[y_kmeans==i,0],X[y_kmeans==i,1],s=100,color=colors[i],label=f'Clusters{i}')
plt.scatter(A_income,S_score,s=150,marker='X',label='New Customer')
plt.title("New Customer Belongs")

plt.legend()
saveimg("NewCustomer.png")
