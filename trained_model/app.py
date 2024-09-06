from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from chat import getResponse

app = Flask(__name__)
CORS(app)  # Set up CORS

@app.route("/")
def index():
    return render_template("api/base.html")

@app.post("/predict")
def predict():
    try:
        text = request.get_json().get("message")
        response = getResponse(text)
        message = {"answer": response}
        return jsonify(message)
    except Exception as e:  # Catch generic exception for robustness
        return jsonify({"error": str(e)}), 400  # Return error response

if __name__ == "__main__":
    app.run(debug=True)
