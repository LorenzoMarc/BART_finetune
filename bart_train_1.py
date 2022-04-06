# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BartTokenizerFast as BartTokenizer, BartModel

from transformers import BartForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
from transformers import BartTokenizerFast as AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

RANDOM_SEED = 42

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

df = pd.read_csv("Politifact_20211230.csv")
df = df[['statement', 'target']]
df = df[df['target'].isin(['false', 'barely-true', 'pants-fire'])]
train_df, val_df = train_test_split(df, test_size=0.2)
MAX_TOKEN_COUNT = 128

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large',num_labels=3)

class BartCondDataset(Dataset):

  def __init__(self, data: pd.DataFrame, tokenizer: BartTokenizer, max_token_len: int = 128):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    statement = data_row.statement
    # if(data_row.false == 1):
    #     label = 1
    # else:
    #     label = 0
    
    label = data_row.target

    input_encodings = self.tokenizer.encode_plus(
      statement,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    target_encodings = self.tokenizer.encode_plus(
      label,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    labels = target_encodings['input_ids']

    return dict(
        input_ids= input_encodings['input_ids'].squeeze(0),
        attention_mask= input_encodings['attention_mask'].squeeze(0),
        labels= labels.squeeze(0)
      )


train_dat = BartCondDataset(
  train_df,
  tokenizer,
  max_token_len=MAX_TOKEN_COUNT
)

test_dat= BartCondDataset(
  val_df,
  tokenizer,
  max_token_len=MAX_TOKEN_COUNT
)


training_args = TrainingArguments(
    output_dir='./models/bart_train',          
    num_train_epochs=5,           
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1,   
    warmup_steps=500,               
    weight_decay=0.01,              
    logging_dir='./logs',          
)

trainer = Trainer(
    model=model,                       
    args=training_args,                  
    train_dataset=train_dat,        
    eval_dataset=test_dat   
)

trainer.train()
