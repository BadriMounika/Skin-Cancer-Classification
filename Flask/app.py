from flask import Flask, request, jsonify
import json
import boto3
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
REGION = os.getenv("REGION")

sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=REGION)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_path = "data/Test/melanoma/ISIC_0000002.jpg"
    file.save(file_path)

    img_array = preprocess_image(file_path)
    payload = json.dumps(img_array.tolist())

    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload
        )
        response_body = response["Body"].read().decode("utf-8")
        result = json.loads(response_body)
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
