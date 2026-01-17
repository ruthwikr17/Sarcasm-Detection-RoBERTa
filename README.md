## Sarcasm Detection using Transformer Models

A complete end-to-end NLP system that detects sarcasm in short-form text using fine-tuned Transformer models.
The project includes model training pipelines, evaluation scripts, and a modern Flask-based web dashboard for real-time sarcasm prediction and model comparison. <br>


## Project Overview

Sarcasm often expresses meaning opposite to literal text, making it difficult for traditional sentiment analysis systems to interpret correctly.
This project builds a context-aware sarcasm detection system by fine-tuning Transformer-based language models to classify tweets as: <br>
	•	Sarcastic <br>
	•	Not Sarcastic <br>

In addition to training high-performing models, the project provides an interactive web dashboard where users can enter any text, select a model, and instantly receive sarcasm predictions with confidence scores. <br>


## Key Features <br>
•	Fine-tuned BERT and RoBERTa models for sarcasm classification <br>
•	Real-time Flask dashboard for single-text prediction <br>
•	Model comparison mode to evaluate BERT vs RoBERTa on the same input <br>
•	Confidence visualization for each prediction <br>
•	Clean, dark-themed UI for demonstration and deployment <br>
•	Modular training, evaluation, and inference scripts <br>


## Technologies Used <br>

**Programming Language** <br>
	•	Python 3.9+ <br>

**Deep Learning & NLP** <br>
	•	PyTorch <br>
	•	Hugging Face Transformers <br>

**Models** <br>
	•	BERT (bert-base-uncased) <br>
	•	RoBERTa (roberta-base) <br>

**Data Processing** <br>
	•	Pandas, NumPy <br>

**Evaluation** <br>
	•	Scikit-learn (Accuracy, Precision, Recall, F1-score) <br>

**Web Deployment** <br>
	•	Flask <br>
	•	HTML, CSS, JavaScript <br>


## Dataset <br>

The models are trained and evaluated on labeled tweet datasets containing sarcastic and non-sarcastic samples. <br>

**Binary Classification Labels:** <br>
	•	1 → Sarcastic <br>
	•	0 → Not Sarcastic <br>

**Data preprocessing includes:** <br>
	•	Cleaning and normalization <br>
	•	Class balancing <br>
	•	Tokenization using model-specific tokenizers <br>
	•	Train-test split to avoid data leakage <br>


## System Workflow

```
Input Tweet
     ↓
Tokenizer (BERT / RoBERTa)
     ↓
Fine-tuned Transformer Model
     ↓
Softmax Classification Layer
     ↓
Prediction + Confidence Score
     ↓
Web Dashboard Output
```


| **Model** | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
| --- | --- | --- | --- | --- |
| RoBERTa | 0.885 | 0.871 | 0.899 | 0.885 |
| BERT | 0.862 | 0.845 | 0.881 | 0.863 |


## Project Structure <br>

```
Sarcasm-Detection-RoBERTa/
│
├── app.py                     
├── templates/                
├── static/                
│
├── Bert.py                   
├── RoBERTa.py          
├── TestBert.py                
├── TestRoberta.py   
├── metrics.py                
├── Cleanup.py             
│
├── Data/                    
├── requirements.txt
└── README.md
```


## Running the Web Dashboard <br>

**Install Dependencies** <br>
```pip install -r requirements.txt```

**Run Flask App** <br>
```python app.py```

**Open in Browser** <br>
```http://127.0.0.1:5000/```
<br>


## Dashboard Capabilities <br>
•	Enter any tweet or sentence <br>
•	Select BERT or RoBERTa model <br>
•	Detect sarcasm in real-time <br>
•	View prediction label and confidence <br>
•	Compare both models on same input <br>

## Example Predictions

**Input:** <br>
“Oh great, another meeting that could’ve been an email. Fantastic.” <br>

**Output:** <br>
Prediction: Sarcastic <br>
Confidence: 0.9997 <br>


## Future Enhancements
•	Multilingual sarcasm detection <br>
•	Larger and more diverse datasets <br>
•	Context-aware conversation-based sarcasm detection <br>

## Author

Ruthvik Reddy
AI / ML Developer
GitHub: https://github.com/ruthwikr17
