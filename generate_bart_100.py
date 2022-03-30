from os import path
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

torch.manual_seed(42)

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

import nltk


filename = 'Politifact_20211230.csv'
# load into a data frame
df = pd.read_csv(filename)

df = df[['statement', 'target']][:30]
df = df[df['target'].isin(['false', 'barely-true', 'pants-fire'])]
df.dropna(inplace=True)  # remove NA values
model_dir = "./models/bart_train/checkpoint-41500"
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

if (path.exists(model_dir)):
    model = BartForConditionalGeneration.from_pretrained(model_dir)
else:
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large', num_labels=3)

MAX_TOKEN_COUNT = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

samples = 10

for num, samp in enumerate(range(samples)):
    print("-----####"+str(num)+"######--------")

    stmt=df.iloc[samp]['statement']
    print("ORIGINAL statement: \n" + stmt)
    nreplace = 3
    words = stmt.split(" ")
    words[nreplace] = "<mask>"
    stmt = " ".join(words)
    print("MASKED statement: \n" + stmt)
    batch = tokenizer(stmt, return_tensors="pt")
    # generated_ids = model.generate(batch["input_ids"],
    #                                do_sample=True,
    #                                top_k=50,
    #                                max_length=300,
    #                                top_p=0.95,
    #                                num_return_sequences=2)
    logits = model(batch["input_ids"])[0]
    masked_index = (batch["input_ids"][0] == tokenizer.mask_token_id).nonzero().item()
    probs = logits[0, masked_index].softmax(dim=0)
    values, predictions = probs.topk(5)
    values = values.detach().numpy()
    splitted = tokenizer.decode(predictions).split()
    for i in range(0,4):
        print((values[i], splitted[i]))
        # res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # print("PREDICTED MASK STATEMENT 1: \n" + res[0])
        # print("PREDICTED MASK STATEMENT 2: \n" + res[1])
        # exit()
