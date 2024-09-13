import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from transformers import AutoModelForSequenceClassification, AutoTokenizer, XLMRobertaForSequenceClassification
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    BertTokenizerFast as BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer)
from datasets import load_dataset
import evaluate

data_files = {"train": "train.csv"}
train_data = load_dataset('/gpfs/home/dcolombaro/Jailbreak_LLM-main/data/jigsaw-toxic-comment-classification-challenge', data_files="train.csv")
eval_data = load_dataset('/gpfs/home/dcolombaro/Jailbreak_LLM-main/data/jigsaw-toxic-comment-classification-challenge', data_files="test_merged.csv")
# train_data['train'] = train_data['train'].select(range(512))
# eval_data['train'] = eval_data['train'].select(range(512))

device = torch.device('cuda')

save_dir='/gpfs/home/dcolombaro/Jailbreak_LLM-main/evaluator/multilingual-toxic-xlm-roberta/fine tuned'
token = '' # add HuggingFace token

print(f"Current device: {device}")

train_data = train_data.remove_columns(['id'])
train_data = train_data.rename_column("comment_text", "text")
eval_data = eval_data.remove_columns(['id'])
eval_data = eval_data.rename_column("comment_text", "text")


def modify_text(example, column_indices):
    example['text'] = example['text'].lower()
    example['text'] = example['text'].replace("\xa0", " ").split()
    example['text'] = ' '.join(example['text'])
    columns = list(example.keys())
    example['label'] = [example[columns[i]] for i in column_indices]
    return example

column_indices=range(1,7)
print("Column indices: ", column_indices)
train_data = train_data.map(lambda x: modify_text(x, column_indices))
eval_data = eval_data.map(lambda x: modify_text(x, column_indices))
columns = list(train_data['train'].column_names)
train_data = train_data.remove_columns(columns[1:7])
print(train_data['train'][1])

MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-05


class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len: int, eval_mode: bool = False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe['train']['text']
        self.eval_mode = eval_mode
        if self.eval_mode is False:
            self.targets = self.data['train']['label']
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        input_ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        output = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_masks': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }

        if self.eval_mode is False:
            output['labels'] = torch.tensor(self.targets[index], dtype=torch.float)

        return output

tokenizer = AutoTokenizer.from_pretrained("unitary/multilingual-toxic-xlm-roberta", truncation=True, do_lower_case=True) # unitary/multilingual-toxic-xlm-roberta
model = AutoModelForSequenceClassification.from_pretrained("unitary/multilingual-toxic-xlm-roberta", num_labels=6, ignore_mismatched_sizes=True)
model = model.to(device)

training_set = MultiLabelDataset(train_data, tokenizer, MAX_LEN)
eval_set = MultiLabelDataset(eval_data, tokenizer, MAX_LEN)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print('Dataset collected: \n', eval_set[0])

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_params = TrainingArguments(
    output_dir=save_dir,
    # evaluation_strategy='epoch',
    # save_strategy='epoch',
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    num_train_epochs=10,
)

metric = evaluate.load('accuracy')

def my_collate(batch):
    # Initialize empty lists for each key
    input_ids = []
    masks = []
    token_type_ids = []
    targets = []

    # Iterate through each element in the batch
    for item in batch:
        print(item)
        input_ids.append(item['input_ids'].squeeze(0))
        masks.append(item['attention_masks'].squeeze(0))
        token_type_ids.append(item['token_type_ids'].squeeze(0))
        targets.append(item['labels'])

    # Stack tensors for each key
    input_ids = torch.stack(input_ids, dim=0)
    masks = torch.stack(masks, dim=0)
    token_type_ids = torch.stack(token_type_ids, dim=0)
    targets = torch.stack(targets, dim=0)

    # Return a dictionary with the collated tensors
    collated = {'input_ids': input_ids, 'attention_masks': masks, 'token_type_ids': token_type_ids, 'labels': targets}
    return collated

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = [item for sublist in labels for item in sublist]
    logits = [item for sublist in logits for item in sublist]
    predictions = np.argmax(logits, axis=-1)
    print(predictions, labels)
    return metric.compute(predictions=logits, references=labels)

trainer = Trainer(
    model=model,
    # data_collator=my_collate,
    args=training_params,
    train_dataset=training_set,
    eval_dataset=eval_set,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained(save_dir, save_config=True)