import torch
from transformers import BlenderbotSmallTokenizer

tokenizer = BlenderbotSmallTokenizer.from_pretrained("./model")
model = torch.load("./model/blenderbotsmall.pt")

UTTERANCE = input("Human: ")

inputs = tokenizer([UTTERANCE], return_tensors="pt")
outputs = model.generate(**inputs)

text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print("Bot: " + text)

REPLY = input("Human: ")

NEXT_UTTERANCE = (UTTERANCE + "</s> <s>" + text + "</s> <s>" + REPLY)

inputs = tokenizer([NEXT_UTTERANCE], return_tensors="pt")
next_reply_ids = model.generate(**inputs)
print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
