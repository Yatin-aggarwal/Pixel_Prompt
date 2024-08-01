from datasets import load_dataset
import numpy as np
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
ds = load_dataset("alfredplpl/anime-with-caption-cc0")

class Dataset(Dataset):
    def __init__(self, transforms=None):
        self.transform = transforms
        self.dataset = ds

    def __len__(self):
        return 6400

    def __getitem__(self, index):
        image = self.dataset["train"][index]["image"]
        prompt = tokenizer(self.dataset["train"][index]["prompt"],add_special_tokens=True,
                           truncation=True, padding=True, return_tensors='pt', max_length=64)
        if self.transform:
            image = self.transform(image)
        return image, prompt["input_ids"]

