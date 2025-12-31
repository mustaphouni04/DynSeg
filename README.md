# DynSeg: Dynamic Separable Convolutions for Zero-Shot Referring Image Segmentation

This repository contains the source code for the DynSeg project.

![Sample image][training_visualizations/example.jpg "Sample prediction in the validation set"]

## Description

Segmentation models rely on fixed parameters but must adapt dynamically to novel concepts defined by free-form text descriptions. Our approach, DynSeg, tackles this challenge by generating task-specific weights directly from language using a hypernetwork. This allows for zero-shot referring image segmentation, where the model can segment objects based on textual descriptions it has never seen during training.

The core of DynSeg is a hypernetwork that takes a text embedding as input and outputs the weights for the convolutional layers of a segmentation network (specifically, a U-Net). This dynamic weight generation allows the segmentation model to adapt on-the-fly to the specific object described in the text.

## Getting Started

### Prerequisites

- Python 3.10 or later
- `uv` package manager (https://github.com/astral-sh/uv)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mustaphouni04/ADV_ML 
   cd DynSeg
   ```

2. Install the dependencies using `uv`:
   ```bash
   uv pip install -e .
   ```
   This command installs the project in editable mode, allowing `uv` to resolve and install all dependencies declared in `pyproject.toml` and pin them in `uv.lock`.

### Training

To run the training script, use the following command:

```bash
uv run python train.py --batch-size <batch_size>
```

Replace `<batch_size>` with your desired batch size.

### Evaluation

To evaluate a trained model, you can use the `evaluate.py` script.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
