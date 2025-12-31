from transformers import CLIPTextModel, AutoTokenizer
from torch import nn
from typing import List

class MultiModalTextEncoder(nn.Module):
    model_name = "openai/clip-vit-large-patch14"

    def __init__(self):
        super().__init__()
        self.model = CLIPTextModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def forward(self, x: str | List[str]):
        x = [x] if isinstance(x, str) else x
        inputs = self.tokenizer(x, padding = True, return_tensors="pt")

        device = self.model.device

        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        return outputs.pooler_output

if __name__ == "__main__":
    text_encoder = MultiModalTextEncoder()

    text = "Hello, my name is Juan"

    output = text_encoder(text)

    print(output.shape)

