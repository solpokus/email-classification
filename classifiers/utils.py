from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # This must be a dict with lists, not pandas Series
        self.labels = labels        # This can be list or numpy array

    def __getitem__(self, idx):
        # encodings values are lists: input_ids[idx], attention_mask[idx], etc.
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# class EmailDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels.reset_index(drop=True)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # Convert input features to tensor
#         item = {
#             key: torch.tensor(self.encodings[key][idx])
#             for key in self.encodings
#         }
#         item['labels'] = torch.tensor(self.labels.iloc[idx])
#         return item


# class EmailDataset(Dataset):
#     def __init__(self, texts, labels):
#         self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128)
#         self.labels = labels

#     def __getitem__(self, idx):
#         return {
#             'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
#             'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
#             # 'labels': torch.tensor(self.labels[idx])
#             'labels': torch.tensor(self.labels.values[idx])
#         }

#     def __len__(self):
#         return len(self.labels)
