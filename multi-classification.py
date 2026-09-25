from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:,:2]
Y = iris.target
logreg = LogisticRegression(solver = 'lbfgs')
logreg.fit(X,Y)

x_min,x_max = X[:,0].min() - .5, X[:,0].max()+ .5

y_min,y_max = X[:,1].min() - .5, X[:,1].max()+ .5
h = .02
xx, yy =np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

z= logreg.predict(np.c_[xx.ravel(),yy.ravel()])

z = z.reshape(xx.shape)
plt.figure(1,figsize=(4,3))

plt.pcolormesh(xx,yy,z,cmap = plt.cm.Paired)

plt.scatter(X[:,0],X[:,1],c=Y,edgecolors='k',cmap=plt.cm.Paired)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
logreg.score(X,Y)

