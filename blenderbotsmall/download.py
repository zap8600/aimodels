import torch
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration

tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
model = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")

tokenizer.save_pretrained("./model")
torch.save(model, "./model/blenderbotsmall.pt")

print("Done!")
