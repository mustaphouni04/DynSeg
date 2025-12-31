import torch
from torch import nn
from transformers import CLIPModel, CLIPImageProcessor
import torch.nn.functional as F


class CLIPBackbone(nn.Module):
    """ CLIP Vision Transformer backbone for image feature extraction.
    use:
        backbone = openai/clip-vit-large-patch14
        features = backbone(images)  # images: (B, 3, H, W) tensor
    """
    def __init__(
            self,
            model_name: str = "openai/clip-vit-large-patch14",
            train_backbone: bool = False,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.vision_model = self.clip.vision_model  # ViT backbone
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)

        self.hidden_size = self.vision_model.config.hidden_size  # D
        self.patch_size = self.vision_model.config.patch_size    # 14
        self.image_size = self.vision_model.config.image_size    # 224

        # opcional: inicialment congelar el backbone
        if not train_backbone:
            for p in self.vision_model.parameters():
                p.requires_grad = False

    def feature_dim(self) -> int:
        """ Returns the dimension of the output features D. """
        return self.hidden_size


    def forward_features(self, pixel_values: torch.Tensor):
        """
        Retorna:
            feat_vec: [B, D] (global)
            feat_map: [B, D, H_p, W_p] (mapa espacial de patches)
        """
        out = self.vision_model(pixel_values=pixel_values)
        tokens = out.last_hidden_state  # [B, N_patches+1, D]

        # token 0 = CLS, la resta són patches
        cls_token = tokens[:, 0]  # [B, D]
        patch_tokens = tokens[:, 1:, :]  # [B, N_patches, D]

        B, N, D = patch_tokens.shape
        Hp = Wp = int(N ** 0.5)  # assumim quadrat: 16x16 per 224x224 amb patch 14

        patch_tokens = patch_tokens.view(B, Hp, Wp, D)  # [B, Hp, Wp, D]
        feat_map = patch_tokens.permute(0, 3, 1, 2).contiguous()  # [B, D, Hp, Wp]

        cls_token = F.normalize(cls_token, dim=-1)
        return cls_token, feat_map  # [B, D], [B, D, Hp, Wp]
    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        out = self.vision_model(pixel_values=pixel_values, output_hidden_states=True)

        hidden_states = out.hidden_states[1:]

        patch_maps = []
        for feats in hidden_states:
            patches = feats[:, 1:]
            B, N, D = patches.shape
            H_p = W_p = int(N ** 0.5)
            patch_map = patches.transpose(1, 2).reshape(B, D, H_p, W_p)
            patch_maps.append(patch_map)

        return patch_maps

    def preprocess_images(self, images, device=None):
        """
        processes a list of PIL images, np arrays or tensors to CLIP format.

        images: image list (PIL, numpy, or similar)
        device: 'cuda', 'cpu', etc. If None, does not do .to(device).
        Returns: pixel_values [B, 3, H, W]
        """
        inputs = self.image_processor(
            images=images,
            return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"]
        if device is not None:
            pixel_values = pixel_values.to(device)

        return pixel_values

    @torch.no_grad()
    def encode_images(self, images, device=None):
        """
        shortcut to process raw images, preprocess and pass through backbone.
        images: PIL images (PIL, numpy, or similar)
        Returns: features [B, D] (no grad)
        """

        pixel_values = self.preprocess_images(images, device=device)
        return self.forward(pixel_values)

    def unfreeze_backbone(self):
        """unfreezes and ready to finetune the backbone."""
        for p in self.vision_model.parameters():
            p.requires_grad = True

    def freeze_backbone(self):
        """freez the backbone again."""
        for p in self.vision_model.parameters():
            p.requires_grad = False
