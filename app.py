from flask import Flask, render_template, request
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name):
    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained("Ruthvik17/bert_sarcasm_model")
        model = BertForSequenceClassification.from_pretrained(
            "Ruthvik17/bert_sarcasm_model"
        )
    else:
        tokenizer = RobertaTokenizer.from_pretrained("Ruthvik17/roberta_sarcasm_model")
        model = RobertaForSequenceClassification.from_pretrained(
            "Ruthvik17/roberta_sarcasm_model"
        )

    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict(tweet, model_name):
    tokenizer, model = load_model(model_name)

    inputs = tokenizer(
        tweet, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    label = "Sarcastic" if pred_class == 1 else "Not Sarcastic"

    model_label = "BERT" if model_name == "bert" else "RoBERTa"

    return label, confidence, model_label


@app.route("/", methods=["GET", "POST"])
def home():
    tweet = ""
    selected_model = "bert"

    result = None
    confidence = None
    model_used = None

    compare_results = None  # for dual model mode

    if request.method == "POST":
        tweet = request.form["tweet"]
        selected_model = request.form["model"]
        action = request.form["action"]

        if action == "detect":
            pred, conf, model_label = predict(tweet, selected_model)
            result = pred
            confidence = f"{conf:.4f}"
            model_used = model_label

        elif action == "compare":
            pred_b, conf_b, _ = predict(tweet, "bert")
            pred_r, conf_r, _ = predict(tweet, "roberta")

            compare_results = {
                "bert_pred": pred_b,
                "bert_conf": f"{conf_b:.4f}",
                "roberta_pred": pred_r,
                "roberta_conf": f"{conf_r:.4f}",
                "better": "BERT" if conf_b > conf_r else "RoBERTa",
            }

    return render_template(
        "index.html",
        tweet=tweet,
        selected_model=selected_model,
        result=result,
        confidence=confidence,
        model_used=model_used,
        compare_results=compare_results,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
