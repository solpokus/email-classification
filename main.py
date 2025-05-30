import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import Trainer, TrainingArguments
from classifiers.model import get_model
from classifiers.metrics import compute_metrics
from classifiers.utils import EmailDataset, tokenizer
from extractors.gpt_extractor import extract_invoice_fields
from transformers import AutoTokenizer
from callbacks import LossHistory
from transformers import AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/sample_emails.csv")
# print(df.columns)  # This will show all column names
print(df["label"].value_counts())
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))


# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", force_download=True)
tokenizer = AutoTokenizer.from_pretrained("roberta-base", force_download=True)


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
# model = get_model(num_labels=len(label_encoder.classes_))
num_classes = len(label_encoder.classes_)
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_classes)

loss_callback = LossHistory()

# Training
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    logging_strategy="epoch",     # log each epoch
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",             # disable wandb/comet/etc.
    save_strategy="no",           # avoid checkpointing for simplicity
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[loss_callback],
    compute_metrics=compute_metrics
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
# sample_email = "Dear Valued Customer,/n Enclosed is your invoice for the period : 202504, please remit full payment immediately before the designated date./n Please drop an email to finance@id.cloud-ace.com should you queries regarding the invoice./n Company Name:/n PT. /n Billing ID:/n 01A4A4/n Invoice number:/n CAI-GCP-2025/n Payment due:/n 31 May 2025/n Important Announcement/n TERM OF PAYMENT RENEWAL (Google Cloud & G Suite)/n As of February 2020, we have renewed out Term of Payment (TOP)'s policy from 30 days to the end of the month (example: invoice dated on February 3rd are to be paid until February 29th)/n Customers who are unable or late to remit payment until the designated date might be at risk for account suspension./n Thank you for trusting Cloud Ace®/n Cloud Ace Integra/n"
sample_email = "Dear RAMA/JUAN MR, \n\nThis is a friendly reminder of your upcoming travel departure.\n\nPhone List:\n628578080 - RAMA/JUAN MR\n628122490 - YUHUDUADUA MR\n628777568 - FADLI MR\n\n\n*Airline*: Vietjet Aviation\n*Flight Number*: VJ874\n*From*:  CGK Soekarno Hatta Intl | Jakarta\n*To*:   AMS Schiphol Arpt | Amsterdam  \n*Departure Date/Time*:  16 May 2025 20:15\n*Arrival Date/Time*:  09 May 2025 07:50\n\nThis information is taken from your booking itinerary and is subject to change.  Please consult directly with your airline for the latest departure information.  \nHave a safe and pleasant flight!\n\nYour friends, at \n==========================="
# sample_email = "Hey there, Welcome to ReadMe! When I started ReadMe, I did it with the belief that API documentation should be the UI/UX for your API. It shouldn’t just tell you how to do things, but rather do as much as it can for you. Similarly, great documentation doesn’t just describe how to do something, it makes it a delightful experience that gets your developers closer to 200s with ease. ReadMe is here to help you build an interactive developer hub that makes the experience for developers using your API not only intuitive but enjoyable, and ensures that you get essential information on your API’s performance. You spent the time tinkering away on your API—now make it shine with ReadMe. Behind every API call is an API user, and we want to help you make their experience great, beginning from the moment they start onboarding. Our secret? Using their real-time request data, ReadMe transforms static docs into a personalized hub that gives your users access to their API keys right from your docs, plus all of their API request history for easy troubleshooting. For your team, our editing experience is designed to let each ReadMe Admin write where they want to write—whether using our MDX-powered editor or in GitHub—so that creating your API docs fits right into your workflow. Whether you work on documentation in ReadMe or your codebase, changes will sync instantly between both. Behind the scenes, visibility into your API usage helps you and your team quickly debug your developers’ requests and understand how your docs are performing, including specific page and user insights. Over the next two weeks, our team will take you on a tour of ReadMe and get you well on your way to launching your developer hub. Stay tuned—there’s lots more good stuff to come. If you have any questions or feedback, or just want to say hello, just reply to this email!"
label = classify_email(sample_email)
print(f"Classified as: {label}")


loss_callback.plot()

# Extract values if it's an invoice
# if label == "invoice":
#     api_key = "your-openai-api-key"
#     extracted = extract_invoice_fields(sample_email, api_key)
#     print("Extracted fields:")
#     print(extracted)
