import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

tokenizer.save_pretrained("./model")
torch.save(model, "./model/distilgpt2.pt")

print("Done!")
