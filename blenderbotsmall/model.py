import torch
from transformers import BlenderbotSmallTokenizer

tokenizer = BlenderbotSmallTokenizer.from_pretrained("./model")
model = torch.load("./model/blenderbotSmall.pt")

UTTERANCE = input("You: ")

inputs = tokenizer([UTTERANCE], return_tensors="pt")
outputs = model.generate(**inputs)

text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print("Bot: " + text)

REPLY = input("You: ")

NEXT_UTTERANCE = (UTTERANCE + "__end__ __start__" + text + "__end__ __start__" + REPLY)

inputs = tokenizer([NEXT_UTTERANCE], return_tensors="pt")
next_outputs = model.generate(**inputs)

next_text = tokenizer.batch_decode(next_outputs, skip_special_tokens=True)[0]
print("Bot: ", next_text)

CONVO = (REPLY + "__end__ __start__" + next_text)

REPLY = input("You: ")

while True:
    CONVO = (CONVO + "__end__ __start__" + REPLY)

    # print("\n" + CONVO + "\n")

    inputs = tokenizer([CONVO], return_tensors="pt")
    next_outputs = model.generate(**inputs)
    
    next_text = tokenizer.batch_decode(next_outputs, skip_special_tokens=True)[0]
    print("Bot: ", next_text)

    CONVO = (REPLY + "__end__ __start__" + next_text)

    # print("\n" + CONVO + "\n")

    REPLY = input("You: ")
