import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deep_supervision import CLIPUNetDeepSupervision
from models.unet.unet_parts import DoubleConv

class HyperNetwork(nn.Module):
    def __init__(self, task_emb_size: int, target_layers: dict[str, nn.Module]):
        super().__init__()
        self.task_emb_size = task_emb_size
        self.heads = nn.ModuleDict()

        for name, layer in target_layers.items():
            if isinstance(layer, nn.Conv2d):
                out_channels = layer.out_channels
                
                sanitized_name = name.replace('.', '_')
                
                self.heads[sanitized_name] = nn.Sequential(
                    nn.Linear(task_emb_size, 256),
                    nn.LayerNorm(256), 
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, out_channels) 
                )
                
                with torch.no_grad():
                    self.heads[sanitized_name][-1].weight.zero_()
                    self.heads[sanitized_name][-1].bias.zero_()

    def forward(self, task_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        generated_weights = {}
        for name, head in self.heads.items():
            task_embedding = F.normalize(task_embedding, dim=1) 
            generated_weights[name] = head(task_embedding)
           
        return generated_weights

class ModulatedUNet(nn.Module):
    def __init__(self, n_classes: int, text_embedding_dim: int=768, modulate_last_outconv_only: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.unet = CLIPUNetDeepSupervision(out_channels=n_classes, num_decoder_layers=4)
        self.modulate_last_outconv_only = modulate_last_outconv_only
        
        self.target_layers = {}
        if self.modulate_last_outconv_only:
            # Target only the last OutConv's conv layer
            if self.unet.heads:
                last_head_idx = len(self.unet.heads) - 1
                last_head_name = f'heads.{last_head_idx}.conv'
                self.target_layers[last_head_name] = self.unet.heads[last_head_idx].conv
        else:
            for name, module in self.unet.named_modules():
                if isinstance(module, DoubleConv):
                    # we are targeting the DoubleConv modules in the decoder part of the U-Net
                    if 'decoders' in name:
                         self.target_layers[name] = module

        self.hypernetwork = HyperNetwork(text_embedding_dim, self.target_layers)

    def forward(self, image: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor | dict:

        modulation_scales = self.hypernetwork(text_embedding)

        features = self.unet.backbone(image)
        x_dec = features[-1]
        input_size = image.shape[2:]
        outputs = []

        for i, decoder in enumerate(self.unet.decoders):
            skip_idx = -2 - i
            if abs(skip_idx) <= len(features):
                skip = features[skip_idx]
            else:
                skip = torch.zeros_like(x_dec)

            x_dec = decoder(x_dec, skip)
            
            
            head = self.unet.heads[i]             
            head_key = f'heads_{i}_conv' 
            
            if head_key in modulation_scales:
                scale = modulation_scales[head_key]
                
                scale = scale.view(scale.shape[0], -1, 1, 1)

                original_weight = head.conv.weight
                
                raw_pred = head.conv(x_dec)
                
                raw_pred = raw_pred * (1 + scale)
                
            else:
                raw_pred = head(x_dec) 
            

            final_pred = F.interpolate(
                raw_pred,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
            outputs.append(final_pred)

        if self.unet.training:
            return outputs
        else:
            return outputs[-1]

if __name__ == '__main__':
    image = torch.randn(1, 3, 256, 256)
    text_embedding = torch.randn(1, 768)

    model = ModulatedUNet(n_classes=1)
    
    image = model.unet.backbone.preprocess_images(image)

    output = model(image, text_embedding)

    if isinstance(output, list):
        print(f"Model returned {len(output)} outputs (deep supervision):")
        for i, tensor in enumerate(output):
            print(f"  Output {i} shape: {tensor.shape}")
    elif isinstance(output, torch.Tensor):
        print(f"Output shape: {output.shape}")
    else:
        print(f"Unexpected output type: {type(output)}")
