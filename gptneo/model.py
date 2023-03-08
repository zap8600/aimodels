import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./model")
model = torch.load("./model/gptneo.pt")

sequence = input("Enter Prompt: ")
max_length_usr = int(input("Max Length of output: "))

inputs = tokenizer.encode(sequence, return_tensors='pt')
outputs = model.generate(inputs, max_length=max_length_usr, do_sample=True)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
