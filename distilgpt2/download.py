import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

tokenizer.save_pretrained("./model")
torch.save(model, "./model/distilgpt2.pt")

print("Done!")
