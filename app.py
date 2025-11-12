from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# --- Load model and vectorizer ---
model_path = os.path.join(os.path.dirname(__file__), 'spam_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Spam Detection API is running 🚀",
        "usage": "POST /predict with JSON: { 'message': 'your text' }"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = data["message"]
    if not message.strip():
        return jsonify({"error": "Empty message"}), 400

    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]
    label = "Not Spam" if prediction == 0 else "Spam"

    return jsonify({
        "message": message,
        "prediction": label
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
