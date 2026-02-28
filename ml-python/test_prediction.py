import joblib
import numpy as np

model = joblib.load("student_model.pkl")


data = np.array([[75, 60, 6, 3]])  


prediction = model.predict(data)

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")