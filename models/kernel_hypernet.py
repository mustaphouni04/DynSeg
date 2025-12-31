import torch
import torch.nn as nn

class KernelHyperNet(nn.Module):
    def __init__(self, task_emb_size: int, target_shapes: dict[str, tuple[int, int]]):
        """
        Hypernetwork that generates convolutional kernels.

        Args:
            task_emb_size (int): The size of the input task embedding.
            target_shapes (dict[str, tuple[int, int]]): A dictionary where keys are target layer names
                                                       and values are tuples of (out_channels, in_channels).
        """
        super().__init__()
        self.task_emb_size = task_emb_size
        self.target_shapes = target_shapes
        self.heads = nn.ModuleDict()
        self.name_map = {}

        for name, (out_channels, in_channels) in target_shapes.items():
            kernel_size = out_channels * in_channels
            sanitized_name = name.replace('.', '_')
            self.name_map[sanitized_name] = name
            
            self.heads[sanitized_name] = nn.Sequential(
                nn.Linear(task_emb_size, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, kernel_size)
            )
            
            with torch.no_grad():
                self.heads[sanitized_name][-1].weight.zero_()
                if self.heads[sanitized_name][-1].bias is not None:
                    self.heads[sanitized_name][-1].bias.zero_()

    def forward(self, task_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Generates the convolutional kernels for the target layers.

        Args:
            task_embedding (torch.Tensor): The task embedding vector.

        Returns:
            dict[str, torch.Tensor]: A dictionary of generated kernels, reshaped to be used in a convolution.
        """
        generated_kernels = {}
        for name, head in self.heads.items():
            original_name = self.name_map[name]
            out_channels, in_channels = self.target_shapes[original_name]
            
            flat_kernel = head(task_embedding)
            kernel = flat_kernel.view(-1, out_channels, in_channels, 1, 1)
            generated_kernels[original_name] = kernel
           
        return generated_kernels