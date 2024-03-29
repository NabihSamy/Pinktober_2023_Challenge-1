import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

file = pd.read_csv('data.csv')

#print (file)

x = file.drop("diagnosis", axis=1)
x=x.drop("Unnamed: 32",axis=1)
x=x.drop("id",axis=1)
#print (x)

y = file["diagnosis"]
Accuracy_max = 0 
index_Max=0
test_size_Max= 0
#print(y)
testSize = 0.1
for i in range(200):
    testSize = 0.1
    while (testSize<0.8):
        #print("i :",i)
        x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=testSize, random_state=i) 

        #print(x_train)
        #print(x_test)
        #print(y_train)
        #print(y_test)

        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        classifier.fit(x_train, y_train)

        #classifier = DecisionTreeClassifier()
        #lassifier.fit(x_train, y_train)

        y_pred= classifier.predict(x_test)
        #print(y_pred)

        cm= confusion_matrix(y_test, y_pred)
        #print(cm)

        accuracy = accuracy_score(y_test, y_pred)
        
        #print(accuracy)
        if Accuracy_max<accuracy:
            test_size_Max = testSize
            Accuracy_max = accuracy
            index_Max = i
            if accuracy==1:
                print("Khlass lkina")
                print(testSize)
                print(i)
        testSize=testSize+0.1

print(test_size_Max)
print(Accuracy_max)
print(index_Max)