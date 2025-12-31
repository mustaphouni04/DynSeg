import torch
from torch import nn
from typing import List
from loguru import logger

class MLPResidualBlock(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, pre_layer_norm, post_dropout
    ):
        super().__init__()
        layers = []
        if pre_layer_norm:
            layers.append(nn.LayerNorm(input_size))
        layers += [
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size, output_size),
            nn.SiLU(),
        ]
        if post_dropout:
            layers.append(nn.Dropout(0.05))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(x)

class UNETModulator(nn.Module):
    def __init__(self, depth_emb_size: int, type_emb_size: int,
                 encoded_task_emb_size: int, model, zero_init_head: bool,
                 head_in_size: int, max_num_layers: int, target_modules: List[nn.Module]):
        super().__init__()

        self.device = ("cuda" if torch.cuda.is_available() else None) or "cpu"
        self.max_num_layers = max_num_layers
        assert self.max_num_layers <= 4, "Number of maximum layers can't be larger than 4"

        self.target_modules = target_modules

        self.layer_depth_encoder = nn.Sequential(
                nn.Embedding(self.max_num_layers, depth_emb_size),
                nn.LayerNorm(depth_emb_size),
            )
        
        self.layer_type_encoder = nn.Sequential(
                nn.Embedding(len(self.target_modules), type_emb_size),
                nn.LayerNorm(type_emb_size),
            )

        self.module_to_int = {m: i for i, m in enumerate(self.target_modules)}

        mlp_inp_size = depth_emb_size + type_emb_size + encoded_task_emb_size

        self.mixer = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size, mlp_inp_size * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size * 4, mlp_inp_size),
            nn.SiLU(),
            nn.Dropout(0.05),
        )

        self.mlp1 = MLPResidualBlock(
            mlp_inp_size,
            mlp_inp_size * 4,
            mlp_inp_size,
            pre_layer_norm=True,
            post_dropout=True,
        )

        self.mlp2 = MLPResidualBlock(
            mlp_inp_size,
            mlp_inp_size * 4,
            mlp_inp_size,
            pre_layer_norm=True,
            post_dropout=True,
        )

        self.mlp3 = nn.Sequential(
            nn.LayerNorm(mlp_inp_size),
            nn.Linear(mlp_inp_size, mlp_inp_size * 4),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(mlp_inp_size * 4, head_in_size),
            nn.SiLU(),
        )

        heads = []
        self.in_features, self.out_features = self.get_in_out_features(model)
        for module in self.target_modules:
            in_features = self.in_features[module]
            out_features = self.out_features[module]

            layer = nn.Linear(
                in_features, out_features, bias=True, device=self.device
            )
            if zero_init_head:
                logger.debug(f"zeroing out head weights for {module}")
                nn.init.zeros_(layer.weight)

            heads.append((module, layer))
        self.heads = nn.ModuleDict(heads)

    def get_in_out_features(self,
        model: nn.Module,
        ) -> tuple[dict[str, int], dict[str, int]]:

        in_features = dict()
        out_features = dict()

        for module_name, module in model.named_modules():
            # this should always pass
            name = module_name.split(".")[-1]

            if name not in in_features:
                in_features[name] = module.in_features
                out_features[name] = module.out_features
            else:
                # assumes each module has the same input and output features
                assert in_features[name] == module.in_features
                assert out_features[name] == module.out_features

        return in_features, out_features


    def _embed_layer_type(self, layer_type) -> torch.Tensor:
        module_idx = self.module_to_int[layer_type]
        module_idx = torch.tensor([module_idx], dtype=torch.long, device=self.device)
        # we only forward one layer type at at time
        # so the shape is always [1, num_target_modules]
        return self.layer_type_encoder(module_idx)

    def _embed_layer_depth(
        self, depth_indices: list[int] | int | torch.Tensor
        ) -> torch.Tensor:
        if isinstance(depth_indices, int):
            depth_indices = torch.tensor(
                [depth_indices], dtype=torch.long, device=self.device
            )
        elif isinstance(depth_indices, list):
            depth_indices = torch.tensor(
                depth_indices, dtype=torch.long, device=self.device
            )

        return self.layer_depth_encoder(depth_indices)

    def _hypernet_forward(self, layer_indices, layer_type, encoded_task_emb):
        # forward one layer type at a time

        bs = len(layer_indices)
        depth_emb = self._embed_layer_depth(layer_indices)  # [bs, depth_emb_size]
        layer_type_emb = self._embed_layer_type(layer_type)  # [1, layer_emb_size]
        layer_type_emb = layer_type_emb.expand(bs, -1)  # [bs, layer_emb_size]
        if encoded_task_emb is None:
            encoded_task_emb = torch.empty(0, device=self.device)

        cat_emb = torch.cat([encoded_task_emb, depth_emb, layer_type_emb], dim=-1)
        mlp_inp = self.mixer(cat_emb)

        mlp_out = self.mlp1(mlp_inp)
        head = self.heads[layer_type]
        head_out = head(self.mlp3(self.mlp2(mlp_out)))
        
        return head_out

    def get_weights(
        self,
        layer_indices: torch.Tensor,
        layer_type: str,
        encoded_task_emb: torch.Tensor = None
    ) -> torch.Tensor:
        
        head_out = self._hypernet_forward(
            layer_indices,
            layer_type,
            encoded_task_emb,
        )

        return head_out 
