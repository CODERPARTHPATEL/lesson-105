import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

digits = load_digits()

x= digits.data
y=digits.target
print('dataset shape',x.shape)
print('target shape',y.shape)
print('classes:',np.unique(y))
plt.imshow(digits.images[0],cmap='grey')
plt.title('example digits')
plt.axis('off')
plt.show()

print('actral digit',y[0])

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    test_size = 0.2,
    random_state=42,
    stratify = y
)
print('traing sample',x_train.shape[0])
print('traing sample',x_test.shape[0])
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print('\nprediction')
print(y_pred[:20])
print('\nprediction')
print(y_test[:20])

accuracy = accuracy_score(y_test,y_pred)
print('\nclassification report')
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)

print('\nconfusion matrix')
print(cm)

plt.figure(figsize=(8,6))
plt.imshow(cm)
plt.title('confusion matrix')
plt.xlabel('predicted label')
plt.ylabel('actal label')
plt.colorbar()
plt.xticks(range(10))
plt.yticks(range(10))
plt.show()
fig, axes = plt.subplots(2,5,figsize=(10,5))

for i,ax in enumerate(axes.ravel()):
    ax.imshow(digits.images[i],cmap='grey')
    ax.set_title(f'actral:{y[i]}')
    ax.axis('off')
    
plt.tight_layout()
plt.show()

sample = x_test[0].reshape(1,-1)
prediction = model.predict(sample)

print('predicted digits',prediction[0])
print('actral digit',y_test[0])

