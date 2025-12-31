import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import CLIPBackbone
from models.unet.unet_parts import OutConv


class HyperLoRA(nn.Module):
    """
    Core Hypernetwork: Fuses Text + Layer ID -> Shared Latent 'z'
    """
    def __init__(self, z_dim=256, num_layers=20, text_emb_dim=768):
        super().__init__()
        self.layer_embedding = nn.Embedding(num_layers, z_dim)
        
        # Project text to z_dim
        self.text_proj = nn.Linear(text_emb_dim, z_dim)
        
        # Residual MLP to mix context
        self.net = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim),
            nn.LayerNorm(z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.LayerNorm(z_dim),
            nn.ReLU()
        )

    def forward(self, text_embedding, layer_idx):
        """
        text_embedding: [B, 768]
        layer_idx: int (scalar) or [B] 
        """
        B = text_embedding.shape[0]
        device = text_embedding.device
        
        # 1. Text Projection
        z_text = self.text_proj(text_embedding) # [B, z_dim]
        
        # 2. Layer Embedding
        # Create tensor for layer_idx if it's just an int
        idx_tensor = torch.tensor(layer_idx, device=device, dtype=torch.long)
        z_layer = self.layer_embedding(idx_tensor) # [z_dim]
        z_layer = z_layer.unsqueeze(0).expand(B, -1) # [B, z_dim]
        
        # 3. Concatenate & Process
        z_in = torch.cat([z_text, z_layer], dim=1) # [B, 2*z_dim]
        z_out = self.net(z_in) # [B, z_dim]
        
        return z_out

class WeightGeneratorHead(nn.Module):
    """
    Project specific head: Latent 'z' -> LoRA Matrices A & B -> Delta W
    """
    def __init__(self, z_dim, rank, c_out, c_in):
        super().__init__()
        self.rank = rank
        self.c_out = c_out
        self.c_in = c_in
        
        self.head_a = nn.Linear(z_dim, c_out * rank)
        self.head_b = nn.Linear(z_dim, rank * c_in)
        
        #nn.init.zeros_(self.head_a.weight)
        #nn.init.zeros_(self.head_a.bias)
        #nn.init.zeros_(self.head_b.weight)
        #nn.init.zeros_(self.head_b.bias)

        nn.init.kaiming_uniform_(self.head_a.weight, a=5**0.5)
        if self.head_a.bias is not None:
            nn.init.zeros_(self.head_a.bias)

        nn.init.zeros_(self.head_b.weight)
        if self.head_b.bias is not None:
            nn.init.zeros_(self.head_b.bias)

    def forward(self, z):
        B = z.shape[0]
        
        mat_a = self.head_a(z).view(B, self.c_out, self.rank)
        mat_b = self.head_b(z).view(B, self.rank, self.c_in)
        
        # Delta = A @ B
        delta_w = torch.bmm(mat_a, mat_b) # [B, c_out, c_in]

        delta_w = torch.tanh(delta_w) * 0.1
        
        # Reshape for Conv2d: [B, Out, In, 1, 1]
        return delta_w.view(B, self.c_out, self.c_in, 1, 1)

# --- SEPARABLE CONV BLOCKS ---

class SeparableConv2d_Hyper(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        # Static Depthwise (Spatial) - Not modulated
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=False)
        
        # Static Pointwise (Channel Mixing) - Modulated
        self.pointwise_static = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x, delta_weight=None):
        out = self.depthwise(x)
        
        if delta_weight is not None:
            # delta_weight shape: [B, Cout, Cin, 1, 1]
            B = delta_weight.shape[0]
            
            # Expand static weight to batch: [B, Cout, Cin, 1, 1]
            w_static = self.pointwise_static.weight.unsqueeze(0).expand(B, -1, -1, -1, -1)
            
            # Combine: W_final = W_static + Delta
            w_final = w_static + delta_weight
            
            # Apply batched convolution using groups=B
            b_size, c, h, w = out.shape
            
            # Reshape input to [1, B*C, H, W] to treat batch as groups
            out_reshaped = out.reshape(1, b_size * c, h, w)
            
            # Reshape weights to [B*Cout, Cin, 1, 1]
            w_final_reshaped = w_final.reshape(b_size * self.pointwise_static.out_channels, c, 1, 1)
            
            # Convolve
            out = F.conv2d(out_reshaped, w_final_reshaped, groups=b_size)
            
            # Reshape output back to [B, Cout, H, W]
            out = out.reshape(b_size, self.pointwise_static.out_channels, h, w)
        else:
            out = self.pointwise_static(out)
            
        return out

class SeparableDoubleConv_Hyper(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.conv1 = SeparableConv2d_Hyper(in_channels, mid_channels)
        #self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn1 = nn.GroupNorm(num_groups=32, num_channels=mid_channels) if mid_channels >= 32 else nn.GroupNorm(num_groups=1, num_channels=mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = SeparableConv2d_Hyper(mid_channels, out_channels)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.GroupNorm(num_groups=32, num_channels=out_channels) if out_channels >= 32 else nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, deltas=None):
        """
        deltas: dict or tuple containing 'conv1' and 'conv2' deltas
        """
        d1 = deltas.get('conv1') if deltas else None
        d2 = deltas.get('conv2') if deltas else None
        
        x = self.conv1(x, delta_weight=d1)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x, delta_weight=d2)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class SeparableCLIPDecoderBlock(nn.Module):
    def __init__(self, in_channels_x1, in_channels_x2, out_channels, scale=2, bilinear=True, use_separable=True):
        super().__init__()
        self.bilinear = bilinear
        self.use_separable = use_separable

        if self.bilinear:
            self.up_x1 = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
            if use_separable:
                self.conv_x1 = SeparableDoubleConv_Hyper(in_channels_x1, in_channels_x1 // 2)
            else:
                self.conv_x1 = nn.Conv2d(in_channels_x1, in_channels_x1 // 2, kernel_size=1) 
        else:
            self.up_x1 = nn.ConvTranspose2d(in_channels_x1, in_channels_x1 // 2, kernel_size=scale, stride=scale)
            self.conv_x1 = nn.Identity()

        if use_separable:
            self.conv_x2 = SeparableDoubleConv_Hyper(in_channels_x2, in_channels_x2 // 2)
        else:
            self.conv_x2 = SeparableDoubleConv_Hyper(in_channels_x2, in_channels_x2 // 2)

        # Final fusion
        self.conv_final = SeparableDoubleConv_Hyper(in_channels_x1 // 2 + in_channels_x2 // 2, out_channels)

    def forward(self, x1, x2, block_deltas=None):
        """
        block_deltas: dict containing {'conv_x1': {...}, 'conv_x2': {...}, 'conv_final': {...}}
        """
        d_x1 = block_deltas.get('conv_x1') if block_deltas else None
        d_x2 = block_deltas.get('conv_x2') if block_deltas else None
        d_final = block_deltas.get('conv_final') if block_deltas else None

        x1_up = self.up_x1(x1)
        
        if self.bilinear and isinstance(self.conv_x1, SeparableDoubleConv_Hyper):
            x1_processed = self.conv_x1(x1_up, deltas=d_x1)
        else:
            x1_processed = self.conv_x1(x1_up)

        x2_resized = F.interpolate(x2, size=x1_up.shape[2:], mode='bilinear', align_corners=False)
        x2_processed = self.conv_x2(x2_resized, deltas=d_x2)

        x_cat = torch.cat([x1_processed, x2_processed], dim=1)
        
        return self.conv_final(x_cat, deltas=d_final)


# --- MAIN MODEL ---

class SeparableCLIPUNet(nn.Module):
    def __init__(self, out_channels=1, num_decoder_layers=4, bilinear=True,
                 layers_to_make_separable=4, # Make all separable for consistency
                 text_embedding_dim=768):
        super().__init__()
        
        self.out_channels = out_channels
        self.n_classes = out_channels
        
        # 1. Backbone
        self.backbone = CLIPBackbone()
        self.backbone.freeze_backbone()

        # 2. Decoder Specs
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
        scale_factors = [first_scale] + [2] * (num_decoder_layers - 1) if num_decoder_layers > 0 else []
        self.active_specs = decoder_specs[:num_decoder_layers]

        self.decoders = nn.ModuleList()
        self.heads = nn.ModuleList([OutConv(spec[2], self.out_channels) for spec in self.active_specs])

        # 3. Build Decoders
        for i, (in_x1, in_x2, out) in enumerate(self.active_specs):
            # We force use_separable=True to enable hypernetwork modulation on all layers
            block = SeparableCLIPDecoderBlock(in_x1, in_x2, out, scale_factors[i], bilinear, use_separable=True)
            self.decoders.append(block)

        # 4. Hypernetwork Setup
        self.z_dim = 256
        self.hyper_rank = 16
        
        # We need a unique ID for every modulated layer.
        # 3 blocks per decoder (x1, x2, final) * 2 convs per block (conv1, conv2) = 6 convs/decoder
        est_total_layers = len(self.decoders) * 6 
        
        self.hyper_core = HyperLoRA(z_dim=self.z_dim, num_layers=est_total_layers + 5, text_emb_dim=text_embedding_dim)
        self.hyper_heads = nn.ModuleDict()
        self.modulated_layers_map = [] # List of (layer_idx, unique_name)

        # 5. Register Heads Automatically
        global_layer_idx = 0
        
        def register_block(block_name, double_conv_module):
            nonlocal global_layer_idx
            # Register Conv1
            name_c1 = f"{block_name}_c1"
            c_in_1 = double_conv_module.conv1.pointwise_static.in_channels
            c_out_1 = double_conv_module.conv1.pointwise_static.out_channels
            self.hyper_heads[name_c1] = WeightGeneratorHead(self.z_dim, self.hyper_rank, c_out_1, c_in_1)
            self.modulated_layers_map.append((global_layer_idx, name_c1))
            global_layer_idx += 1
            
            # Register Conv2
            name_c2 = f"{block_name}_c2"
            c_in_2 = double_conv_module.conv2.pointwise_static.in_channels
            c_out_2 = double_conv_module.conv2.pointwise_static.out_channels
            self.hyper_heads[name_c2] = WeightGeneratorHead(self.z_dim, self.hyper_rank, c_out_2, c_in_2)
            self.modulated_layers_map.append((global_layer_idx, name_c2))
            global_layer_idx += 1

        for i, decoder in enumerate(self.decoders):
            # Decoder i
            if isinstance(decoder.conv_x1, SeparableDoubleConv_Hyper):
                register_block(f"dec_{i}_x1", decoder.conv_x1)
            
            register_block(f"dec_{i}_x2", decoder.conv_x2)
            register_block(f"dec_{i}_final", decoder.conv_final)

    def forward(self, x, text_embedding):
        # 1. Backbone Features
        features = self.backbone(x)
        x_dec = features[-1]
        input_size = x.shape[2:]

        # 2. Generate Hypernetwork Deltas
        # We generate a flat dict first: {'dec_0_x1_c1': delta, ...}
        flat_deltas = {}
        for layer_idx, name in self.modulated_layers_map:
            head = self.hyper_heads[name]
            # Get shared latent z
            z = self.hyper_core(text_embedding, layer_idx) # [B, z_dim]
            # Get delta
            delta = head(z) # [B, Cout, Cin, 1, 1]
            flat_deltas[name] = delta

        # 3. Structure Deltas for Decoders
        # We need to pack them into: decoder_deltas[i] = {'conv_x1': {'conv1': d, 'conv2': d}, ...}
        
        def get_block_deltas(prefix):
            c1 = flat_deltas.get(f"{prefix}_c1")
            c2 = flat_deltas.get(f"{prefix}_c2")
            if c1 is None and c2 is None: return None
            return {'conv1': c1, 'conv2': c2}

        outputs = []
        for i, decoder in enumerate(self.decoders):
            skip_idx = -2 - i
            skip = features[skip_idx] if abs(skip_idx) <= len(features) else torch.zeros_like(x_dec)

            # Assemble deltas for this decoder block
            block_deltas = {
                'conv_x1': get_block_deltas(f"dec_{i}_x1"),
                'conv_x2': get_block_deltas(f"dec_{i}_x2"),
                'conv_final': get_block_deltas(f"dec_{i}_final")
            }

            # Forward Decoder
            x_dec = decoder(x_dec, skip, block_deltas=block_deltas)

            # Head & Interpolate
            raw_pred = self.heads[i](x_dec)
            
            # Interpolate to input size 
            final_pred = F.interpolate(raw_pred, size=input_size, mode='bilinear', align_corners=False)
            outputs.append(final_pred)

        if self.training:
            return outputs
        else:
            return outputs[-1]
