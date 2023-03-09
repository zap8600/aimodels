import torch
from transformers import CodeGenTokenizer, CodeGenForCausalLM

tokenizer = CodeGenTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = CodeGenForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

tokenizer.save_pretrained("./model")
torch.save(model, "./model/codegen.pt")

print("Done!")
