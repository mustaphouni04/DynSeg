import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.deep_supervision import CLIPUNetDeepSupervision
from models.unet.unet_parts import DoubleConv

class RandLoRAHyperNetwork(nn.Module):
    def __init__(self, task_emb_size: int, target_layers: dict[str, nn.Module], rank: int = 8):
        super().__init__()
        self.heads = nn.ModuleDict()
        self.rank = rank

        self.frozen_bases = nn.ParameterDict()

        for name, layer in target_layers.items():
            layers_to_process = []
            if isinstance(layer, nn.Conv2d):
                layers_to_process.append((name, layer))
            elif isinstance(layer, DoubleConv):
                layers_to_process.append((f"{name}_conv1", layer.double_conv[0]))
                layers_to_process.append((f"{name}_conv2", layer.double_conv[3]))

            for sub_name, conv_layer in layers_to_process:
                if not isinstance(conv_layer, nn.Conv2d): continue

                safe_name = sub_name.replace('.', '_')

                out_dim = conv_layer.out_channels
                in_dim = conv_layer.in_channels * conv_layer.kernel_size[0] * conv_layer.kernel_size[1]

                u_rand = torch.randn(out_dim, rank) / math.sqrt(rank)
                v_rand = torch.randn(rank, in_dim) / math.sqrt(in_dim)

                self.register_buffer(f"{safe_name}_U", u_rand)
                self.register_buffer(f"{safe_name}_V", v_rand)

                head = nn.Sequential(
                    nn.Linear(task_emb_size, 128), 
                    nn.SiLU(),                     
                    nn.Linear(128, rank)           
                )
                self.heads[safe_name] = head

    def forward(self, task_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        generated_weights = {}

        for name, head in self.heads.items():
            sigma_pred = head(task_embedding)

            U = getattr(self, f"{name}_U") 
            V = getattr(self, f"{name}_V") 

            batch_size = sigma_pred.shape[0]
            U_expanded = U.unsqueeze(0).expand(batch_size, -1, -1) 
            V_expanded = V.unsqueeze(0).expand(batch_size, -1, -1) 

            sigma_matrix = torch.diag_embed(sigma_pred) 

            weight_flat = torch.bmm(torch.bmm(U_expanded, sigma_matrix), V_expanded)

            generated_weights[name] = weight_flat

        return generated_weights

class ModulatedUNet(nn.Module):
    def __init__(self, n_classes: int, text_embedding_dim: int=768):
        super().__init__()
        self.unet = CLIPUNetDeepSupervision(out_channels=n_classes, num_decoder_layers=4)
        self.n_classes = n_classes
        self.n_channels = 3

        self.target_layers = {}
        for name, module in self.unet.named_modules():
            if isinstance(module, DoubleConv):
                if 'decoders' in name or 'outc' in name:
                     self.target_layers[name] = module

        self.hypernetwork = RandLoRAHyperNetwork(text_embedding_dim, self.target_layers, rank=8)

    def forward(self, image: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor | list:
        if text_embedding.dim() == 3:
            text_embedding = text_embedding.squeeze(1)

        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        generated_weights = self.hypernetwork(text_embedding)

        # TODO: replace F.conv2d with a batched functional call.

        batch_size = image.shape[0]

        for name, module in self.target_layers.items():
            sanitized_name = name.replace('.', '_')

            w_flat = generated_weights.get(f"{sanitized_name}_conv1")
            if w_flat is not None:
                conv1 = module.double_conv[0]

                # bs = 1
                w_reshaped = w_flat[0].view(conv1.weight.shape)
            
                if isinstance(conv1.weight, nn.Parameter):
                    del conv1.weight
                conv1.weight = w_reshaped

            w_flat = generated_weights.get(f"{sanitized_name}_conv2")
            if w_flat is not None:
                conv2 = module.double_conv[3]
                w_reshaped = w_flat[0].view(conv2.weight.shape)

                if isinstance(conv2.weight, nn.Parameter):
                    del conv2.weight
                conv2.weight = w_reshaped

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
            raw_pred = self.unet.heads[i](x_dec)
            final_pred = F.interpolate(raw_pred, size=input_size, mode='bilinear', align_corners=False)
            outputs.append(final_pred)

        if self.unet.training:
            return outputs
        else:
            return outputs[-1]

if __name__ == '__main__':
    print("--- Initializing ModulatedUNet with RandLoRA ---")
    
    n_classes = 1
    batch_size = 1 
    image_size = 256
    text_emb_dim = 768
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    try:
        model = ModulatedUNet(n_classes=n_classes, text_embedding_dim=text_emb_dim).to(device)
        print("Model created successfully.")
    except NameError as e:
        print(f"\nError: Could not instantiate model. Ensure 'models.deep_supervision' is in your python path.\nDetails: {e}")
        exit()

    dummy_image = model.unet.backbone.preprocess_images(torch.rand(batch_size, 3, image_size, image_size).to(device))
    dummy_text_emb = torch.rand(batch_size, text_emb_dim).to(device)
    
    print("\nRunning Forward Pass...")
    try:
        output = model(dummy_image, dummy_text_emb)
        
        if isinstance(output, list):
            print(f"Success! Model returned {len(output)} deep supervision outputs.")
            print(f"Final output shape: {output[-1].shape}")
        elif isinstance(output, torch.Tensor):
            print(f"Success! Output shape: {output.shape}")
        else:
            print(f"Unexpected output type: {type(output)}")
            
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--- RandLoRA Efficiency Check ---")
        print(f"Total Parameters:     {total:,}")
        print(f"Trainable Parameters: {trainable:,}")
        print(f"Trainable Ratio:      {trainable/total:.2%}")
        
    except Exception as e:
        print(f"\nForward pass failed: {e}")
        import traceback
        traceback.print_exc()
