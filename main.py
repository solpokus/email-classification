import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import Trainer, TrainingArguments
from classifiers.model import get_model
from classifiers.utils import EmailDataset, tokenizer
from extractors.gpt_extractor import extract_invoice_fields
from transformers import AutoTokenizer
from callbacks import LossHistory
import torch
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/sample_emails.csv")
# print(df.columns)  # This will show all column names
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", force_download=True)


encodings = tokenizer(
    # list(df["text"]),  # or df["email_body"]
    list(df["email_body"]),
    truncation=True,
    padding=True,
    max_length=512,
    # return_tensors=None  # important: don't return tensors yet
    return_tensors="pt"  # important: don't return tensors yet
)

# Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    list(df["email_body"]), list(df["label_encoded"]), test_size=0.2, random_state=42
)

# Tokenize splits
train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=512
)
test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=512
)

# Prepare datasets
train_dataset = EmailDataset(train_encodings, train_labels)
test_dataset = EmailDataset(test_encodings, test_labels)

# Model
model = get_model(num_labels=len(label_encoder.classes_))

loss_callback = LossHistory()

# Training
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    evaluation_strategy="epoch",
    logging_strategy="epoch",     # log each epoch
    logging_dir="./logs",
    report_to="none",             # disable wandb/comet/etc.
    save_strategy="no",           # avoid checkpointing for simplicity
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[loss_callback]
)

trainer.train()


# Test a classification
def classify_email(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs).item()
    return label_encoder.inverse_transform([pred_idx])[0]

# sample_email = "Please see attached invoice INV-991 for the month of March."
# sample_email = "Hi Preflight Team,/n Please take action for Preflight Information"
sample_email = "Dear RAMA/JUAN MR, \n\nThis is a friendly reminder of your upcoming travel departure.\n\nPhone List:\n6285780809136 - RAMA/JUAN MR\n6281224900871 - YUHUDUADUA MR\n6287775681352 - FADLI MR\n\n\n*Airline*: Vietjet Aviation\n*Flight Number*: VJ874\n*From*:  CGK Soekarno Hatta Intl | Jakarta\n*To*:   AMS Schiphol Arpt | Amsterdam  \n*Departure Date/Time*:  16 May 2025 20:15\n*Arrival Date/Time*:  09 May 2025 07:50\n\nThis information is taken from your booking itinerary and is subject to change.  Please consult directly with your airline for the latest departure information.  \nHave a safe and pleasant flight!\n\nYour friends, at \n==========================="
label = classify_email(sample_email)
print(f"Classified as: {label}")


loss_callback.plot()

# Extract values if it's an invoice
# if label == "invoice":
#     api_key = "your-openai-api-key"
#     extracted = extract_invoice_fields(sample_email, api_key)
#     print("Extracted fields:")
#     print(extracted)
