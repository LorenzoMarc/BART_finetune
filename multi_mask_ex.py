from transformers import BartTokenizer, BartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
TXT = "My friends are <mask> but they eat too many carbs."

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
# logits = model(input_ids).logits
#
# masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
# probs = logits[0, masked_index].softmax(dim=0)
# values, predictions = probs.topk(5)

summary_ids = model.generate(input_ids, num_beams=4, max_length=5)
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
# res = tokenizer.decode(predictions).split()
# print(res)

# from transformers import BartTokenizer, BartModel
#
# import torch
# model_name='facebook/bart-large'
# tokenizer = BartTokenizer.from_pretrained(model_name)
#
# model = BartModel.from_pretrained(model_name)
#
# sentence = "A chart shows China and <mask> are leading in per capita <mask> emissions."
#
# token_ids = tokenizer.encode(sentence, return_tensors='pt')
#
# # print(token_ids)
#
# token_ids_tk = tokenizer.tokenize(sentence, return_tensors='pt')
#
# print(token_ids_tk)
#
# masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
#
# masked_pos = [mask.item() for mask in masked_position]
#
# print(masked_pos)
#
# with torch.no_grad():
#     output = model(token_ids)
#
# last_hidden_state = output[0].squeeze()
#
# print("\n\n")
#
# print("sentence : ", sentence)
#
# print("\n")
#
# list_of_list = []
#
# for mask_index in masked_pos:
#     mask_hidden_state = last_hidden_state[mask_index]
#
#     idx = torch.topk(mask_hidden_state, k=10, dim=0)[1]
#
#     words = [tokenizer.decode(i.item()).strip() for i in idx]
#
#     list_of_list.append(words)
#
#     print(words)
#
# best_guess = ""
#
# for j in list_of_list:
#     best_guess = best_guess + " " + j[0]