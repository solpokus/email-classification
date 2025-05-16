from transformers import BertForSequenceClassification

def get_model(num_labels):
    return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
