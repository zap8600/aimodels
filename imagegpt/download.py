import torch
from transformers import AutoImageProcessor, ImageGPTForCausalImageModeling

image_processor = AutoImageProcessor.from_pretrained('openai/imagegpt-small')
model = ImageGPTForCausalImageModeling.from_pretrained('openai/imagegpt-small')

image_processor.save_pretrained("./model")
torch.save(model, "./model/imagegpt.pt")

print("Done!")
