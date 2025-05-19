from transformers import BertForSequenceClassification
from transformers import AutoModelForSequenceClassification

def get_model(num_labels):
    # return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    return AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=num_classes)
