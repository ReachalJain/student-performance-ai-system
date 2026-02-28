import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


data = pd.read_excel("student_data.xlsx")


data["final_result"] = data["final_result"].map({"Pass":1, "Fail":0})


X = data[["Attendance","Test","Assignments","Study_hours"]]
y = data["final_result"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LogisticRegression()
model.fit(X_train, y_train)


joblib.dump(model, "student_model.pkl")

print("Model trained successfully 🎉")