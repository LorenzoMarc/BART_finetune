# -*- coding: utf-8 -*-

from os import path
import pandas as pd
from tabulate import tabulate
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

masked = 'masked_samples.csv'
original = 'samples_statements.csv'
# load into a data frame
df_masked = pd.read_csv(masked)
df_ori = pd.read_csv(original)
length = len(df_masked)
print(length)

# model_dir = "./models/bart_train/checkpoint-41500"
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# if (path.exists(model_dir)):
#     model = BartForConditionalGeneration.from_pretrained(model_dir)
# else:
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

with open('predicted_multi_masked_token.txt', 'w', ) as f:

    for num, samp in enumerate(range(length-1)):
        f.write("\n\n-----####"+str(num+1)+"######--------\n")
        print("-----####"+str(num+1)+"######--------")

        stmt = df_ori.iloc[samp]['statement']
        f.write("-----####ORIGINAL statement: \n" + stmt+"\n")
        print("ORIGINAL statement: \n" + stmt)
        msk = df_masked.iloc[samp]['statement']

        f.write("MASKED statement: \n" + msk+"\n")
        print("MASKED statement: \n" + msk)

        batch = tokenizer(msk, return_tensors="pt")
        generated_ids = model.generate(batch["input_ids"],
                                       max_new_tokens=35,# or this,not both: max_length=300,
                                       # do_sample=True,
                                       top_k=5,
                                       top_p=0.9,
                                       # temperature=0.9,
                                       # num_beams=6,
                                       no_repeat_ngram_size=3,
                                       num_return_sequences=1,
                                       # repetition_penalty=1.3,
                                       min_length=50
                                       # early_stopping=True
                                       )
        res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print("GENERATED statement: \n")
        for s in range(len(res)):
            print(str(res[s]) + "\n")
        f.write("GENERATED statement: \n" )
        for s in range(len(res)):
            f.write(str(res[s]) + "\n")
f.close()

exit()
#         logits = model(batch["input_ids"])[0]
#         masked_index = (batch["input_ids"][0] == tokenizer.mask_token_id).nonzero()
#         masked_pos = [mask.item() for mask in masked_index]
#         # print(masked_pos)
#         words = []
#         pred_values = []
#         for i in masked_pos:
#             probs = logits[0, i].softmax(dim=0)
#             # print(probs)
#             values, predictions = probs.topk(1)
#             values = values.cpu()
#             values = values.detach().numpy()
#             pred_values.append(values)
#             # print(values)
#
#             splitted = tokenizer.decode(predictions).split()
#             words.append(splitted)
#             # print(splitted)
#             res = batch['input_ids'][0]
#             res[i] = predictions
#         res = tokenizer.decode(res, skip_special_tokens=True)
#         print("GENERATED statement: \n" + res)
#
#         f.write(tabulate({'Word': words, 'Prediction': pred_values}, headers="keys"))
#         for i in range(len(words)):
#             # f.write("word(n.{:}): {:}, pred: {:}\n".format(i+1, values[i], splitted[i]))
#             print((pred_values[i][0], words[i]))
# f.close()
#
#         # for i in range(len(values)):
#         # dict = {'Word': splitted, 'Prediction': values}
#
#
# # for num, samp in enumerate(range(length-1)):
# #     print("-----####"+str(num)+"######--------")
# #
# #     stmt=df_ori.iloc[samp]['statement']
# #     print("ORIGINAL statement: \n" + stmt)
# #     msk = df_masked.iloc[samp]['statement']
# #     print("MASKED statement: \n" + msk)
# #     batch = tokenizer(msk, return_tensors="pt")
# #     generated_ids = model.generate(batch["input_ids"],
# #                                    do_sample=True,
# #                                    top_k=50,
# #                                    max_length=300,
# #                                    top_p=0.95,
# #                                    num_return_sequences=1)
# #     res = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# #     print("GENERATED statement: \n" + res+ "\n")
#
#     # logits = model(batch["input_ids"])[0]
#     # masked_index = (batch["input_ids"][0] == tokenizer.mask_token_id).nonzero().item()
#     # probs = logits[0, masked_index].softmax(dim=0)
#     # values, predictions = probs.topk(5)
#     # values = values.detach().numpy()
#     # splitted = tokenizer.decode(predictions).split()
#     # for i in range(0,4):
#     #     print((values[i], splitted[i]))
#         # res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#         # print("PREDICTED MASK STATEMENT 1: \n" + res[0])
#         # print("PREDICTED MASK STATEMENT 2: \n" + res[1])
#         # exit()
