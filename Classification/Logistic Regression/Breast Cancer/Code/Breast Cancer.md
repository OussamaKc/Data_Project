# Logistic Regression

## Importing the libraries


```python
import pandas as pd
```

## Importing the dataset


```python
dataset = pd.read_csv('Desktop/machine Learning/Machine+Learning+A-Z+(Codes+and+Datasets)/Machine Learning A-Z (Codes and Datasets)\Part 3 - Classification/Section 14 - Logistic Regression/Python/case study/Final Folder/Dataset/breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
```

## Splitting the dataset into the Training set and Test set


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

## Training the Logistic Regression model on the Training set


```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
```




    LogisticRegression(random_state=0)



## Predicting the Test set results


```python
y_pred = classifier.predict(X_test)
print(y_pred)
```

    [2 2 4 4 2 2 2 4 2 2 4 2 4 2 2 2 4 4 4 2 2 2 4 2 4 4 2 2 2 4 2 4 4 2 2 2 4
     4 2 4 2 2 2 2 2 2 2 4 2 2 4 2 4 2 2 2 4 4 2 4 2 2 2 2 2 2 2 2 4 4 2 2 2 2
     2 2 4 2 2 2 4 2 4 2 2 4 2 4 4 2 4 2 4 4 2 4 4 4 4 2 2 2 4 4 2 2 4 2 2 2 4
     2 2 4 2 2 2 2 2 2 2 4 2 2 4 4 2 4 2 4 2 2 4 2 2 4 2]
    

## Making the Confusion Matrix


```python
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_pred, y_test))
accuracy_score(y_pred, y_test)
```

    [[84  3]
     [ 3 47]]
    




    0.9562043795620438



## Computing the accuracy with k-Fold Cross Validation


```python
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
```

    Accuracy: 96.70 %
    Standard Deviation: 1.97 %
    


```python

```
