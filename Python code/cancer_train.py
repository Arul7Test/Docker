from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# Load dataset
data = load_breast_cancer()
# Organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
# Look at our data
print(label_names)
print(labels)
print(feature_names)
print(features)

# Split our data
train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)



# Initialize our classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(train, train_labels)
# Make predictions
preds = gnb.predict(test)
print(preds)

# Evaluate accuracy
print(accuracy_score(test_labels, preds))

X = np.array([[1000,2000],
             [10001,8500],
             [2500,4500],
             [15400,8800],
             [9500,7500],
             [12000,8700]])

y = [0,1,0,1,0,1]

clf = svm.SVC(kernel='linear', C = 1.0 )
clf.fit(X,y)
Z = np.array([[7,2]])
print(clf.predict(Z))
Z = np.array([ [ 7,15 ] ] )
print(clf.predict(Z))


w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,16000)
yy = a * xx - clf.intercept_[0] / w[1]

print( xx )
print( yy )

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()



