from flask import Flask, request
import pickle   

#initialise the Flask app
app = Flask(__name__)


@app.route("/hello", methods=["GET"])
def foo():
    return "<H3> Hello User welcome to MLOPs class 5Feb25 2348"


if __name__ == "__main__":
    app.run(debug=True, port=5004)

# call classifier.py from app.py to get prediction and leverage post method to get the data from user


