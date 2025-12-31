import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F
import math

class CoreHyperNetwork(nn.Module):
    def __init__(self, f_size=1, z_dim=64, out_size=16, in_size=16):
        super(CoreHyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        
        self.layer1 = nn.Linear(z_dim, in_size * z_dim)
        #self.layer1 = utils.spectral_norm(nn.Linear(z_dim, in_size * z_dim))
        
        self.layer2 = nn.Linear(z_dim, out_size * f_size * f_size)
        #self.layer2 = utils.spectral_norm(nn.Linear(z_dim, out_size * f_size * f_size))
        
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.layer1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.layer1.bias, 0)

        nn.init.normal_(self.layer2.weight, mean=0, std=0.001)
        nn.init.constant_(self.layer2.bias, 0)

    def forward(self, z):
        B = z.shape[0]

        h_in = self.layer1(z)
        
        h_in = h_in.view(B, self.in_size, self.z_dim)
        
        h_in = torch.nn.functional.elu(h_in) 
        #h_in = F.leaky_relu(h_in, 0.2)

        h_final = self.layer2(h_in) # (B, in_size, out_size*f*f)
        
        kernel = h_final.view(B, self.in_size, self.out_size, self.f_size, self.f_size)
        
        kernel = kernel.permute(0, 2, 1, 3, 4)

        return kernel

class HyperLoRA(nn.Module):
    """
    Generates Low-Rank adaptations (A, B) for a weight matrix W.
    W_new = W_static + (A @ B)
    
    This drastically reduces the complexity of the manifold the hypernetwork has to learn.
    """
    def _init_(self, z_dim=256, rank=8, num_layers=10):
        super()._init_()
        self.z_dim = z_dim
        self.rank = rank
        
        # 1. Layer Embeddings: The network learns "which layer am I modulating?"
        self.layer_embedding = nn.Embedding(num_layers, z_dim)
        
        # 2. The Core Generator (Shared across all layers)
        # We use a simple residual MLP
        self.net = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim), # Input: Text Emb (768) proj to z_dim + Layer Emb
            nn.LayerNorm(z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.LayerNorm(z_dim),
            nn.ReLU()
        )
        
        # 3. Text Projection (To match z_dim)
        self.text_proj = nn.Linear(768, z_dim)

    def forward(self, text_embedding, layer_idx, shape_out, shape_in):
        """
        text_embedding: [B, 768]
        layer_idx: [1] (Integer index of the layer)
        """
        B = text_embedding.shape[0]
        
        # 1. Prepare Context
        z_text = self.text_proj(text_embedding) # [B, z_dim]
        z_layer = self.layer_embedding(torch.tensor(layer_idx, device=z_text.device)) # [z_dim]
        
        # Expand layer embedding to batch size
        z_layer = z_layer.unsqueeze(0).expand(B, -1)
        
        # Concatenate
        z_in = torch.cat([z_text, z_layer], dim=1) # [B, 2*z_dim]
        
        # 2. Generate Latent Code
        z_out = self.net(z_in) # [B, z_dim]
        
        # 3. Generate Low-Rank Matrices A and B
        # We need to generate A: [B, shape_out, rank] and B: [B, rank, shape_in]
        
        # We use dynamic linear layers (or just massive linear projections if shapes vary)
        # Since shapes vary per layer, we can't have a single head.
        # But we can have a hypernet output a flat vector and reshape it.
        
        # NOTE: For efficiency in a "Universal" model, usually we have specific heads 
        # for specific shapes, OR we generate a large stream and slice it. 
        # Here, I'll use a specific head generated on the fly or stored in a ModuleDict in the main class.
        # But to keep this class clean, let's assume this class returns the 'z_out' 
        # and the specific projections happen in the main model.
        
        return z_out

class WeightGeneratorHead(nn.Module):
    """
    Project specific head for each layer size.
    """
    def _init_(self, z_dim, rank, c_out, c_in):
        super()._init_()
        self.rank = rank
        self.c_out = c_out
        self.c_in = c_in
        
        # Head A: z -> c_out * rank
        self.head_a = nn.Linear(z_dim, c_out * rank)
        
        # Head B: z -> rank * c_in
        self.head_b = nn.Linear(z_dim, rank * c_in)
        
        # Initialize to ZERO. This is crucial for the "Identity" start.
        # The generated delta will be 0 at the start of training.
        nn.init.zeros_(self.head_a.weight)
        nn.init.zeros_(self.head_a.bias)
        nn.init.zeros_(self.head_b.weight)
        nn.init.zeros_(self.head_b.bias)

    def forward(self, z):
        B = z.shape[0]
        
        mat_a = self.head_a(z).view(B, self.c_out, self.rank)
        mat_b = self.head_b(z).view(B, self.rank, self.c_in)
        
        # Compute Delta W = A @ B
        delta_w = torch.bmm(mat_a, mat_b) # [B, c_out, c_in]
        
        # Reshape for Conv2d (c_out, c_in, 1, 1)
        return delta_w.view(B, self.c_out, self.c_in, 1, 1)

class TextConditionedEmbedding(nn.Module):
    def __init__(self, target_shape, z_dim, chunk_shape, task_emb_size):
        super(TextConditionedEmbedding, self).__init__()
        self.out_channels, self.in_channels = target_shape
        self.chunk_out, self.chunk_in = chunk_shape
        self.z_dim = z_dim
        
        self.h = self.out_channels // self.chunk_out
        self.k = self.in_channels // self.chunk_in
        
        self.z_list = nn.ParameterList()
        num_chunks = self.h * self.k
        #total_chunks = self.h * self.k # +

        #all_zs = torch.empty(total_chunks, self.z_dim) # +
        #nn.init.orthogonal_(all_zs) # +
        #for i in range(total_chunks): # +
            #self.z_list.append(nn.Parameter(all_zs[i])) # +

        for _ in range(num_chunks):
            self.z_list.append(nn.Parameter(torch.randn(self.z_dim)))

        self.text_proj = nn.Sequential(
            nn.Linear(task_emb_size, z_dim),
            nn.LayerNorm(z_dim), 
            nn.ReLU()
        )

        """
        self.text_proj = nn.Sequential(
            nn.Linear(task_emb_size, z_dim * 2),
            nn.LayerNorm(z_dim * 2), 
            nn.ReLU(),
            nn.Linear(z_dim * 2, z_dim)
        )
        
        """
        self.ln_z = nn.LayerNorm(z_dim)

    def forward(self, hyper_net, text_embedding):
        z_text = self.text_proj(text_embedding) # (B, z_dim)
        
        ww = [] 
        for i in range(self.h):
            w = [] 
            for j in range(self.k):
                z_static = self.z_list[i * self.k + j] 
                
                # Combine and Normalize
                z_combined = z_static.unsqueeze(0) + z_text 
                z_combined = self.ln_z(z_combined)
                
                kernel_chunk = hyper_net(z_combined)
                w.append(kernel_chunk)
            
            ww.append(torch.cat(w, dim=2))
        
        full_kernel = torch.cat(ww, dim=1)
        
        return full_kernel
