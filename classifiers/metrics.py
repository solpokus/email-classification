from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    # preds = p.predictions.argmax(-1)
    # labels = p.label_ids
    # return {
    #     'accuracy': accuracy_score(labels, preds),
    #     'f1': f1_score(labels, preds, average='weighted')
    # }
