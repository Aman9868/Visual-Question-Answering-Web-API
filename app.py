from flask import render_template,request,redirect,url_for, flash,jsonify
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
from flask import Flask
import os
import cv2
import torch
import tensorflow as tf
app = Flask(__name__)
app.config['OUTPUT_FOLDER'] = 'static/output'
@app.route('/',methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        image.save(os.path.join(app.config['OUTPUT_FOLDER'], 'input_image.jpg'))
        image=Image.open(image)
        print(image)
        vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        text = request.form["question"]
        # prepare inputs
        encoding = vilt_processor(image, text, return_tensors="pt")
        # forward pass
        outputs = vilt_model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        ans=vilt_model.config.id2label[idx]
        print(ans)
        ps="Visual Question Answering"
        return render_template("answer.html", answer=ans,purpose=ps)
    return render_template("index.html")
app.run(debug=True)