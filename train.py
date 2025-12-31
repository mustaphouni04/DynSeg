import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import toml

import wandb
from evaluate import evaluate
from models.deep_supervision import CLIPUNetDeepSupervision 
from models.separable_unet import SeparableCLIPUNet
from models.modulated_unet import ModulatedUNet
import albumentations as A
from models.text_backbone import MultiModalTextEncoder 
from albumentations.pytorch.transforms import ToTensorV2
from datasets import load_dataset
from utils.dataset import Box2SegmentDataset, Poly2SegmentDataset
from utils.dice_score import dice_loss
from src.visualization import save_training_image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1.0,
        amp: bool = True,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
    ):

    try:

        config = toml.load("configs/basic.toml")
        ds = load_dataset(config["utils"]["dataset_name"], split="train")
        
        target_size = config["utils"]["target_size"]

        transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=35, p=1.0),
            A.Normalize(mean=CLIP_MEAN, std=CLIP_STD, max_pixel_value=255.0),
            ToTensorV2(), ])

        dataset = Poly2SegmentDataset(ds, config, transform=transform)

    except (AssertionError, RuntimeError, IndexError):
        raise ValueError("Give a proper dataset!!!")

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    experiment = wandb.init(project='U-Net Hypernetwork', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    #optimizer = optim.AdamW(model.parameters(),
                              #lr=learning_rate, weight_decay=weight_decay, foreach=True)

    hypernet_params = list(map(id, model.hyper_core.parameters())) + \
                  list(map(id, model.hyper_heads.parameters()))
    base_params = filter(lambda p: id(p) not in hypernet_params, model.parameters())

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': learning_rate},
        {'params': model.hyper_core.parameters(), 'lr': learning_rate * 2}, # Boost Hypernet
        {'params': model.hyper_heads.parameters(), 'lr': learning_rate * 2}
    ], weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    text_encoder = MultiModalTextEncoder().to(device)
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                img, mask, txt, img_id  = batch
                with torch.no_grad():
                    txt = text_encoder(txt)

                images = img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = mask.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_preds = model(images, txt)
                    layer_weights = [0.1, 0.2, 0.3, 1.0]

                    total_loss = 0

                    for i, pred in enumerate(masks_preds):
                        pred_f32 = pred.float() 
                        true_masks_float = true_masks.float() 
                        bce = criterion(pred_f32.squeeze(1), true_masks_float)
                        pred_prob = F.sigmoid(pred_f32.squeeze(1))
                        dice = dice_loss(pred_prob, true_masks_float, multiclass=False)
                        d_score = 1 - dice
                        loss = 0.2*bce + 0.8*dice
                        experiment.log({f'layer_{i+1} loss': loss.item()})
                        experiment.log({f'layer_{i+1} dice': dice.item()})
                        experiment.log({f'layer_{i+1} dice score': d_score.item()})
                        experiment.log({f'layer_{i+1} bce': bce.item()})
                        total_loss += layer_weights[i] * loss

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(total_loss).backward()
                grad_scaler.unscale_(optimizer)
                grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]))
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += total_loss.item()
                experiment.log({
                    'train loss': total_loss.item(),
                    'grad_norm': grad_norm.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': total_loss.item()})

                if global_step > 0 and global_step % 25 == 0:
                    with torch.no_grad():
                        pred_for_viz = F.sigmoid(masks_preds[-1])
                        dataset.save_prediction(pred_for_viz[-1], int(img_id[-1]),
                            save_path=f"training_visualizations/step_0.png",
                                                thresh=0.5)

                division_step = (n_train // (2 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if value.grad is not None:
                                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_preds[-1].argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

                if save_checkpoint and global_step % 2100 == 0:
                    dir_checkpoint = "checkpoints"
                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                    state_dict = model.state_dict()
                    #state_dict['mask_values'] = dataset.mask_values
                    torch.save(state_dict, str(dir_checkpoint + '/' + 'checkpoint_step_{}_epoch_{}.pth'.format(global_step, epoch)))
                    logging.info(f'Checkpoint {epoch} saved!')


        dir_checkpoint = "checkpoints"
        torch.save(model.state_dict, str(dir_checkpoint + '/' + 'checkpoint_epoch{}.pth'.format(epoch)))

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #model = CLIPUNetDeepSupervision(out_channels=1, num_decoder_layers=4)
    #model = ModulatedUNet(n_classes=1)
    model = SeparableCLIPUNet()
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t 3 input channels\n'
                 f'\t 1 output channels (classes)\n')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        #del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
