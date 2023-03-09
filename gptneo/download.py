import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

tokenizer.save_pretrained("./model")
torch.save(model, "./model/gptneo.pt")

print("Done!")
