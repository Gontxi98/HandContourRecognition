import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score,classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree

def predict_gesture(HandLandmark):
    

     data = pd.read_csv("./HandContourRecognition/Dataset/unified.csv",index_col=0)

     X = data.drop("Gesto",axis="columns").values
     Y = data["Gesto"].values

     X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

     clf = tree.DecisionTreeClassifier()

     clf = clf.fit(X_train,Y_train)

     # fig = plt.figure(figsize=(25,20))
     # tree.plot_tree(clf, feature_names=data.drop("Gesto", axis='columns').columns, filled=True)

     print('Accuracy of decision tree classifier on training set: {:.2f}'
          .format(clf.score(X_train, Y_train)))
     print('Accuracy of decision tree classifier on test set: {:.2f}'
          .format(clf.score(X_test, Y_test)))

     print("Accuracy in test set:")
     y_pred = clf.predict(HandLandmark)

     # print("Error cuadrático medio (MSE): ", mean_squared_error(Y_test, y_pred))
     # print("Estadístico R_2: ", r2_score(Y_test, y_pred))

     return y_pred