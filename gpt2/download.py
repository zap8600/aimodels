import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

tokenizer.save_pretrained("./model")
torch.save(model, "./model/gpt2.pt")

print("Done!")
