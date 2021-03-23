import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


df = pd.read_csv("diabetes.csv")
#print(df.head())


train, test = train_test_split(df, test_size=0.2, stratify=df['Outcome'])

x_train = train[train.columns[:8]]
x_test = test[test.columns[:8]]
y_train = train['Outcome']
y_test = test['Outcome']

logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

print("Accuracy Score is :", metrics.accuracy_score(y_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)

plt.figure()
plt.matshow(confusion_matrix, cmap='Pastel1')

for x in range(0, 2):
    for y in range(0, 2):
        plt.text(x, y, confusion_matrix[x, y])
        
plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.show()
