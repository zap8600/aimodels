import torch
from transformers import BlenderbotTokenizer

tokenizer = BlenderbotTokenizer.from_pretrained("./model")
model = torch.load("./model/blenderbot.pt")

UTTERANCE = input("You: ")

inputs = tokenizer([UTTERANCE], return_tensors="pt")
outputs = model.generate(**inputs)

text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print("Bot: " + text)

REPLY = input("You: ")

NEXT_UTTERANCE = (UTTERANCE + "</s> <s>" + text + "</s> <s>" + REPLY)

inputs = tokenizer([NEXT_UTTERANCE], return_tensors="pt")
next_outputs = model.generate(**inputs)

next_text = tokenizer.batch_decode(next_outputs, skip_special_tokens=True)[0]
print("Bot: ", next_text)

CONVO = (REPLY + "</s> <s>" + next_text)

REPLY = input("You: ")

while True:
    CONVO = (CONVO + "</s> <s>" + REPLY)

    # print("\n" + CONVO + "\n")

    inputs = tokenizer([CONVO], return_tensors="pt")
    next_outputs = model.generate(**inputs)
    
    next_text = tokenizer.batch_decode(next_outputs, skip_special_tokens=True)[0]
    print("Bot: ", next_text)

    CONVO = (REPLY + "</s> <s>" + next_text)

    # print("\n" + CONVO + "\n")

    REPLY = input("You: ")
