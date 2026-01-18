from flask import Flask, render_template, request
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification

app = Flask(__name__)

DEVICE = torch.device("cpu")

bert_tokenizer = BertTokenizer.from_pretrained("Ruthvik17/bert_sarcasm_model")
bert_model = BertForSequenceClassification.from_pretrained(
    "Ruthvik17/bert_sarcasm_model", dtype=torch.float32, low_cpu_mem_usage=False
)
bert_model.eval()

roberta_tokenizer = RobertaTokenizer.from_pretrained("Ruthvik17/roberta_sarcasm_model")
roberta_model = RobertaForSequenceClassification.from_pretrained(
    "Ruthvik17/roberta_sarcasm_model",
    dtype=torch.float32,
    low_cpu_mem_usage=False,
)
roberta_model.eval()


def load_model(model_name):
    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained("Ruthvik17/bert_sarcasm_model")
        model = BertForSequenceClassification.from_pretrained(
            "Ruthvik17/bert_sarcasm_model",
            dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
    else:
        tokenizer = RobertaTokenizer.from_pretrained("Ruthvik17/roberta_sarcasm_model")
        model = RobertaForSequenceClassification.from_pretrained(
            "Ruthvik17/roberta_sarcasm_model",
            dtype=torch.float32,
            low_cpu_mem_usage=False,
        )

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

    return label, confidence


@app.route("/", methods=["GET", "POST"])
def home():
    tweet = ""
    selected_model = "bert"

    result = None
    confidence = None
    model_used = None
    compare_results = None

    if request.method == "POST":
        tweet = request.form["tweet"]
        selected_model = request.form["model"]
        action = request.form["action"]

        if action == "detect":
            pred, conf = predict(tweet, selected_model)
            result = pred
            confidence = f"{conf:.4f}"
            model_used = selected_model.upper()

        elif action == "compare":
            pred_b, conf_b = predict(tweet, "bert")
            pred_r, conf_r = predict(tweet, "roberta")

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
        model_name=model_used,
        compare_results=compare_results,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
