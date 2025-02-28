from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "Pere-ere/fine_tuned_bert_film_sentiment_modell"

# Load model & tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model loaded successfully from Hugging Face!")


text = "This movie was absolutely amazing! The storytelling was brilliant."

inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print(outputs.logits)  # Check raw model predictions
