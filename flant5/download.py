import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

tokenizer.save_pretrained("./model")
torch.save(model, "./model/flant5.pt")

print("Done!")
