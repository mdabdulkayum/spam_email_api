from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# --- Load model and vectorizer ---
model_path = os.path.join(os.path.dirname(__file__), 'spam_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    print("ERROR LOADING MODEL:", e)
    model = None
    vectorizer = None

# --- Routes ---
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"error": "Empty message"}), 400

        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)[0]
        label = "Spam" if prediction else "Not Spam"

        # Probability
        proba = model.predict_proba(message_vector)[0]  # returns [not_spam_prob, spam_prob]

        probability = float(proba[1]) if label == "Spam" else float(proba[0])

        # probability_data = {
        #     "not_spam": float(proba[0]),
        #     "spam": float(proba[1])
        # }

        return jsonify({
            "message": message,
            "prediction": label,
            "probability": probability
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": f"Error predicting: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
