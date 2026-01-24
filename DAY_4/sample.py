import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

X_train = np.array([[158, 58], [158, 59], [158, 63], [160, 59], [160, 60], 
                    [163, 60], [163, 61], [160, 64], [163, 64], [165, 61], 
                    [165, 62], [165, 65], [168, 62], [168, 63], [168, 66], 
                    [170, 63], [170, 64], [170, 68]])
y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
try:
    h=int(input("Enter the height:"))
    w=int(input("Enter the weight:"))
except TypeError:
    print("Enter the valid numbers")
    exit()
predict=model.predict([[h,w]])
size="large" if predict[0]==1 else "small"
print(f"Prediction :{size}")

plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap='coolwarm',s=50,label='Training Data')
color='red' if predict[0]==1 else 'blue'
plt.scatter(h,w,c=color,s=50,marker='*',label='New Data Point')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')   
plt.title('KNN Classification of Size based on Height and Weight')
plt.legend()
plt.show()