from sklearn.datasets import load_iris
import mlflow
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def train_model_with_base_and_new_data(base_data, new_data):
    # Step 1: Load the Iris dataset from data folder using pd.read_csv
    df = pd.read_csv(base_data)
    df_new = pd.read_csv(new_data)
    # update df with new data
    df = pd.concat([df, df_new], ignore_index=True)

    print(df.head())
    
    # Step 2: Split the data into training and testing sets
    # drop Species and ID from X
    X = df.drop(columns=["Species", "Id"])

    Y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    Y_pred = clf.predict(X_test)
    

    print(f"Accuracy: {accuracy_score(y_test, Y_pred)}")

    #pickle the model
    with open("./model/iris_classfier.pkl", "wb") as f:
        pickle.dump(clf, f)

train_model_with_base_and_new_data("data/iris.csv", "data/New_Iris.csv")

