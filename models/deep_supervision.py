import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import CLIPBackbone
from models.unet.unet_parts import DoubleConv, OutConv
import toml

class CLIPDecoderBlock(nn.Module):
    def __init__(self, in_channels_x1, in_channels_x2, out_channels, scale=2, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        self.scale = scale

        if self.bilinear:
            self.up_x1 = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
            self.conv_x1 = DoubleConv(in_channels_x1, in_channels_x1 // 2)
        else:
            self.up_x1 = nn.ConvTranspose2d(in_channels_x1, in_channels_x1 // 2, kernel_size=scale, stride=scale)
            self.conv_x1 = nn.Identity()

        self.conv_x2 = DoubleConv(in_channels_x2, in_channels_x2 // 2)
        self.conv_final = DoubleConv(in_channels_x1 // 2 + in_channels_x2 // 2, out_channels)

    def forward(self, x1, x2):
        x1_up = self.up_x1(x1)
        x1_processed = self.conv_x1(x1_up) if self.bilinear else x1_up
        
        x2_resized = F.interpolate(x2, size=x1_up.shape[2:], mode='bilinear', align_corners=False)
        x2_processed = self.conv_x2(x2_resized)

        x_cat = torch.cat([x1_processed, x2_processed], dim=1)
        return self.conv_final(x_cat)


class CLIPUNetDeepSupervision(nn.Module):
    def __init__(self, out_channels=1, num_decoder_layers=4, bilinear=True):
        super(CLIPUNetDeepSupervision, self).__init__()
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.backbone = CLIPBackbone()
        self.backbone.freeze_backbone()

        decoder_specs = [
            (1024, 1024, 512),
            (512,  1024, 256),
            (256,  1024, 128),
            (128,  1024, 64),
        ]

        MAX_LAYERS = 4

        num_decoder_layers = min(num_decoder_layers, MAX_LAYERS)
        
        missing_stages = MAX_LAYERS - num_decoder_layers
        first_scale = 2 ** (missing_stages + 1)

        if num_decoder_layers > 0:
            scale_factors = [first_scale] + [2] * (num_decoder_layers - 1)
        else:
            scale_factors = []

        self.active_specs = decoder_specs[:num_decoder_layers]

        self.decoders = nn.ModuleList([
                CLIPDecoderBlock(in_x1, in_x2, out, scale_factors[i], bilinear)
                for i, (in_x1, in_x2, out) in enumerate(self.active_specs)
        ])

        self.heads = nn.ModuleList([
            OutConv(spec[2], self.out_channels) for spec in self.active_specs
        ])

    def forward(self, x):
        features = self.backbone(x)
        x_dec = features[-1]
        
        input_size = x.shape[2:] 
        
        outputs = []

        for i, decoder in enumerate(self.decoders):
            skip_idx = -2 - i # maybe * instead of -
            
            if abs(skip_idx) <= len(features):
                skip = features[skip_idx]
            else:
                skip = torch.zeros_like(x_dec) 

            x_dec = decoder(x_dec, skip)

            raw_pred = self.heads[i](x_dec)
            
            final_pred = F.interpolate(
                raw_pred, 
                size=input_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            outputs.append(final_pred)

        if self.training:
            return outputs
        else:
            return outputs[-1]

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = toml.load("configs/basic.toml")
    model = CLIPUNetDeepSupervision(out_channels=1, 
                                    num_decoder_layers=config["unet"]["num_decoder_layers"],
                                    bilinear=config["unet"]["bilinear"]).to(device)
    
    dummy_image = torch.rand((1, 3, 684, 454)).to(device)
    
    outputs = model.backbone.preprocess_images(dummy_image)
    outputs = model(outputs)
    
    print(f"Training Mode: {model.training}")
    if isinstance(outputs, list):
        print(f"Number of outputs (Deep Supervision): {len(outputs)}")
        for idx, out in enumerate(outputs):
            print(f"Output {idx} shape: {out.shape}")
    else:
        print(f"Final output shape: {outputs.shape}")
