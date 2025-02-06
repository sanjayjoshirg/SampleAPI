from flask import Flask, request, jsonify
import pickle
import numpy
from classifierPost import train_model_with_base_and_new_data
import json

#initialise the Flask app
app = Flask(__name__)   

#import the pickle model
with open("./model/iris_classfier.pkl", "rb") as f:
    clf = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json
    print(payload)
    X_unknown = payload["SepalLengthCm"], payload["SepalWidthCm"], payload["PetalLengthCm"], payload["PetalWidthCm"]
    X_unknown = numpy.array(X_unknown).reshape(1, -1)
    print(X_unknown)
    prediction = clf.predict(X_unknown)
    return jsonify({"prediction": str(prediction)})
    return None


# call the train_model_with_base_and_new_data function from classifierPost.py
@app.route("/train", methods=["POST"])
def train():
    payload = request.json
    print(payload)
    train_model_with_base_and_new_data(payload["base_data"], payload["new_data"])
    return jsonify({"message": "Model trained successfully"})

if __name__ == "__main__":
    app.run(port=5005)

# call classifier.py from app.py to get prediction and leverage post method to get the data from user


