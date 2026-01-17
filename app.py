from flask import Flask, render_template, request
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT
bert_tokenizer = BertTokenizer.from_pretrained("bert_sarcasm_model")
bert_model = BertForSequenceClassification.from_pretrained("bert_sarcasm_model")
bert_model.to(DEVICE)
bert_model.eval()

# Load RoBERTa
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta_sarcasm_model")
roberta_model = RobertaForSequenceClassification.from_pretrained(
    "roberta_sarcasm_model"
)
roberta_model.to(DEVICE)
roberta_model.eval()


def predict(tweet, model_name):
    if model_name == "bert":
        tokenizer = bert_tokenizer
        model = bert_model
        model_label = "BERT"
    else:
        tokenizer = roberta_tokenizer
        model = roberta_model
        model_label = "RoBERTa"

    inputs = tokenizer(
        tweet, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    prediction = "Sarcastic" if pred_class == 1 else "Not Sarcastic"

    return prediction, confidence, model_label


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
    app.run(debug=True)
