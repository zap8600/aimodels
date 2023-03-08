import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

tokenizer.save_pretrained("./model")
torch.save(model, "./model/flant5.pt")

print("Done!")
