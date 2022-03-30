# -*- coding: utf-8 -*-

from tabulate import tabulate

# Commented out IPython magic to ensure Python compatibility.
import os
import time
import datetime
from os import path
import pandas as pd
import seaborn as sns
import numpy as np
import random

import matplotlib.pyplot as plt
# % matplotlib inline

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

torch.manual_seed(42)

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

import nltk


filename = 'Politifact_20211230.csv'
# load into a data frame
df = pd.read_csv(filename)

df = df[['statement', 'target']][:400]
df = df[df['target'].isin(['false', 'barely-true', 'pants-fire'])]
df.dropna(inplace=True)  # remove NA values
output_dir = "./model_save-bart/"

if (path.exists(output_dir)):
    tokenizer = BartTokenizer.from_pretrained(output_dir)
    model = BartForConditionalGeneration.from_pretrained(output_dir)
else:
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
MAX_TOKEN_COUNT = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class BartCondDataset(Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer: BartTokenizer, max_token_len:int=MAX_TOKEN_COUNT):
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

        label = data_row.statement

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
            input_ids=input_encodings['input_ids'].squeeze(0),
            attention_mask=input_encodings['attention_mask'].squeeze(0),
            labels=labels.squeeze(0)
        )


dataset = BartCondDataset(
  df,
  tokenizer,
  max_token_len=MAX_TOKEN_COUNT
)

# Split into training and validation sets
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

batch_size = 2

# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order.
train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler=RandomSampler(train_dataset),  # Select batches randomly
    batch_size=batch_size  # Trains with this batch size.
)

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
    val_dataset,  # The validation samples.
    sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
    batch_size=batch_size  # Evaluate with this batch size.
)


model.to(device)

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# some parameters I cooked up that work reasonably well

epochs = 5
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

# this produces sample output every 100 steps
sample_every = 100

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
optimizer = AdamW(model.parameters(),
                  lr=learning_rate,
                  eps=epsilon
                  )

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=total_steps)


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


total_t0 = time.time()

training_stats = []

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch['input_ids'].to(device)
        b_masks = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        model.zero_grad()

        outputs = model(b_input_ids,
                        labels=b_labels,
                        attention_mask=b_masks
                        )

        loss = outputs[0]

        batch_loss = loss.item()
        total_train_loss += batch_loss

        # # Get sample every x batches.
        # if step % sample_every == 0 and not step == 0:
        #
        #     elapsed = format_time(time.time() - t0)
        #     print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader),
        #                                                                              batch_loss, elapsed))
        #
        #     model.eval()
        #
        #     sample_outputs = model.generate(b_input_ids,
        #                                     do_sample=True,
        #                                     top_k=50,
        #                                     max_length=200,
        #                                     top_p=0.95,
        #                                     num_return_sequences=1
        #                                     )
        #     for i, sample_output in enumerate(sample_outputs):
        #         print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
        #
        #     model.train()

        loss.backward()

        optimizer.step()

        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        b_input_ids = batch['input_ids'].to(device)
        b_masks = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids,
                            #                            token_type_ids=None,
                            attention_mask=b_masks,
                            labels=b_labels)

            loss = outputs[0]

        batch_loss = loss.item()
        total_eval_loss += batch_loss

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# A hack to force the column headers to wrap.
# df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

# Display the table.
print(df_stats)

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12, 6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4, 5])

plt.savefig(output_dir + 'loss.png')
plt.close()
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BART-LARGE model has {:} different named parameters.\n'.format(len(params)))


print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))

samples=10
with open(output_dir + 'predicted_masked_token.txt', 'a') as f:

    for num, samp in enumerate(range(samples)):
        f.write("\n\n-----####"+str(num+1)+"######--------\n")
        print("-----####"+str(num+1)+"######--------")

        stmt=df.iloc[samp]['statement']
        f.write("-----####ORIGINAL statement: \n" + stmt+"\n")
        print("ORIGINAL statement: \n" + stmt)
        nreplace = 3
        words = stmt.split(" ")
        words[nreplace] = "<mask>"
        stmt = " ".join(words)
        f.write("MASKED statement: \n" + stmt+"\n")
        print("MASKED statement: \n" + stmt)
        batch = tokenizer(stmt, return_tensors="pt").to(device)
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
        values = values.cpu()
        values = values.detach().numpy()
        splitted = tokenizer.decode(predictions).split()

        # for i in range(len(values)):
        # dict = {'Word': splitted, 'Prediction': values}
        f.write(tabulate({'Word': splitted, 'Prediction': values}, headers="keys"))
        for i in range(0,4):
            # f.write("word(n.{:}): {:}, pred: {:}\n".format(i+1, values[i], splitted[i]))
            print((values[i], splitted[i]))
f.close()
# mask = "<mask>"
#
# sample = RandomSampler(dataset)
# batch = tokenizer(sample, return_tensors="pt")
# generated_ids = model.generate(batch["input_ids"])
# # assert tok.batch_decode(generated_ids, skip_special_tokens=True) == [
# #     "UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria"
# # ]
# res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# print(res)
#
# generated = torch.tensor(tokenizer.encode(mask)).unsqueeze(0)
# generated = generated.to(device)
#
# print(generated)

# sample_outputs = model.generate(
#     generated,
#     # bos_token_id=random.randint(1,30000),
#     do_sample=True,
#     top_k=50,
#     max_length=300,
#     top_p=0.95,
#     num_return_sequences=10
# )


