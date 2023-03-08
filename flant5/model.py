import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./model")
model = torch.load("./model/flant5.pt")

task = input("Task: ")
prompt = input("Enter Prompt: ")
sequence = task + ": " + prompt
max_length_usr = int(input("Max Length of output: "))

inputs = tokenizer(sequence, return_tensors='pt').input_ids
outputs = model.generate(inputs, max_length=max_length_usr, do_sample=True)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
