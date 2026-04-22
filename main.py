from flask import Flask, render_template, request, send_from_directory, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import logging
from transformers import pipeline
from dotenv import load_dotenv
import cv2

app = Flask(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if HF_TOKEN is None:
    raise ValueError("❌ HuggingFace token not found in .env")


generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    token=HF_TOKEN
)


model = load_model("model/model.h5")
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']


upload_folder = "./uploads"
os.makedirs(upload_folder, exist_ok=True)

os.makedirs("static/results", exist_ok=True)

last_prediction = {"label": None, "confidence": None}


def temperature_scaling(probs, T=1.5):
    log_probs = np.log(probs + 1e-10)
    scaled = log_probs / T
    exp = np.exp(scaled)
    return exp / np.sum(exp)

def predict_tumor(image_path):
    img = load_img(image_path, target_size=(224,224))
    img = img_to_array(img)

    # Correct preprocessing
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Raw prediction
    raw_preds = model.predict(img)[0]

    # Optional: apply temperature scaling
    preds = temperature_scaling(raw_preds, T=1.5)

    idx = np.argmax(preds)
    confidence = float(preds[idx])
    label = class_labels[idx]

    # 🔥 IMPORTANT: decision logic
    if confidence < 0.65:
        result = "Invalid or unclear MRI image"
    elif label == "notumor":
        result = "No Tumor"
    else:
        result = f"Tumor: {label}"

    return result, confidence, label


def get_status(label, confidence):
    if label == "notumor":
        return "Not dangerous"
    elif confidence >= 0.85:
        return "Dangerous (high confidence)"
    else:
        return "Uncertain, consult doctor"



def detect_intent(msg):
    msg = msg.lower()

    if any(x in msg for x in ["result", "condition", "prediction"]):
        return "result"

    elif any(x in msg for x in ["danger", "serious", "safe", "risk"]):
        return "danger"

    elif any(x in msg for x in ["what should i do", "next", "treatment", "solution"]):
        return "next_step"

    elif any(x in msg for x in ["hi", "hello", "hey"]):
        return "greeting"

    elif "bye" in msg:
        return "bye"

    else:
        return "explanation"


def generate_explanation(label, confidence, user_input):

    confidence_percent = f"{confidence*100:.2f}%"
    status = get_status(label, confidence)

    prompt = f"""
You are a medical AI assistant for MRI brain tumor results.

Your job:
Explain the result clearly to a non-technical person.

Data:
- Condition: {label}
- Confidence: {confidence_percent}
- Status: {status}

User question: {user_input}

Strict Rules:
- Answer in 2–3 short sentences
- No medical jargon
- Be calm and supportive
- If tumor → clearly say consult doctor
- If no tumor → reassure
- Add one short positive line
- Do NOT ask questions
- Do NOT explain AI/model

Answer:
"""

    response = generator(
        prompt,
        max_new_tokens=100,
        temperature=0.3
    )

    return response[0]["generated_text"].strip()


def handle_chat(user_msg):

    intent = detect_intent(user_msg)

    label = last_prediction["label"]
    confidence = last_prediction["confidence"]

    if not label:
        return "Please upload an MRI image first."

    # RESULT
    if intent == "result":
        return f"Detected Condition: {label}\nModel Confidence: {confidence*100:.2f}%"

    # DANGER
    elif intent == "danger":
        return get_status(label, confidence)

    # NEXT STEP
    elif intent == "next_step":
        if label == "notumor":
            return "No immediate action needed. Maintain a healthy lifestyle and routine checkups."
        else:
            return "You should consult a neurologist or medical professional for further diagnosis."

    # GREETING
    elif intent == "greeting":
        return "Hello! I can help you understand your MRI result."

    # BYE
    elif intent == "bye":
        return "Take care. Stay healthy."

    # DEFAULT → LLM
    else:
        return generate_explanation(label, confidence, user_msg)


def create_result_image(image_path, label, confidence):
    img = cv2.imread(image_path)

    text = f"{label} ({confidence*100:.2f}%)"

    cv2.putText(
        img,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    result_filename = "result_" + os.path.basename(image_path)
    result_path = os.path.join("static/results", result_filename)

    cv2.imwrite(result_path, img)

    return result_filename


# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")

        if not file:
            return render_template("index.html", result="No file uploaded")

        path = os.path.join(upload_folder, file.filename)
        file.save(path)

        result, confidence, label = predict_tumor(path)

        last_prediction["label"] = label
        last_prediction["confidence"] = confidence

        result_filename = create_result_image(path, label, confidence)

        return render_template(
            "index.html",
            result=result,
            confidence=f"{confidence*100:.2f}%",
            file_path=f"/uploads/{file.filename}"
        )

    return render_template("index.html", result=None)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_msg = data.get("message")

        response = handle_chat(user_msg)

        return jsonify({"response": response})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"response": "Server error"}), 500



@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(upload_folder, filename)

if __name__ == "__main__":
    app.run(debug=True)