import os
import logging
import argparse
import torch
import torch.optim
import torch.utils.data
from torch.optim import lr_scheduler
from utils.dataloader import get_data_loader, get_graph_and_word_file
from utils.checkpoint import load_pretrained_model
from src.loss_functions.vpu_loss import vpuLoss
from src.helper_functions.helper_functions import ModelEma
from src.model.Backbone import Backbone
from src.model.Global_Branch import convert_to_lgconv

# Argument Parsing
parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training on CPU')
parser.add_argument('data', metavar='DIR', help='path to dataset', nargs='?', default=r"C:\\PU-MLC-main\\datasets")
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='resnet01')
parser.add_argument('--model-path', type=str)
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
parser.add_argument('--thre', default=0.8, type=float, metavar='N')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
parser.add_argument('--print-freq', '-p', default=64, type=int, metavar='N')
parser.add_argument('--dataset', default='COCO2014')
parser.add_argument('--prob', default=0.5, type=float)
parser.add_argument('--cropSize', default=448, type=int, metavar='N')
parser.add_argument('--scaleSize', default=512, type=int, metavar='N')
parser.add_argument('--gamma', type=float, metavar='N')
parser.add_argument('--topK', default=1, type=int, metavar='N')
parser.add_argument('--pretrainedModel', type=str)
parser.add_argument('--alpha', type=float, metavar='N')
parser.add_argument('--ema', type=float, metavar='N')
parser.add_argument('--Stop_epoch', type=int, metavar='N', default=None)

# Logger Setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def fix_bn(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()

# Set device to CPU
device = torch.device("cpu")
print("Running on CPU...")

def main():
    args = parser.parse_args()
    args.do_bottleneck_head = False

    os.makedirs("exp/log", exist_ok=True)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_path = f'exp/log/train_{str(args.prob)}.log'
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    train_loader, val_loader, sampler = get_data_loader(args)
    GraphFile, WordFile = get_graph_and_word_file(args, train_loader.dataset.changedLabels)

    print('Creating model...')
    model = Backbone(GraphFile, WordFile, classNum=args.num_classes, topK=args.topK)

    if args.pretrainedModel and args.pretrainedModel != 'None':
        print("Loading pretrained model...")
        model = load_pretrained_model(model, args)

    print('Model created!')

    # Convert to lgconv if necessary
    convert_to_lgconv(model.resnet)  # Change here to use self.resnet instead of self.backbone

    args.mix_alpha = 0.3
    model.to(device)

    weight_decay = 1e-4
    Epochs = 80

    # Ensure 'alpha' has a valid value (default to 1.0 if not provided)
    alpha = args.alpha if args.alpha is not None else 1.0
    gamma = args.gamma if args.gamma is not None else 0.0  # Default to 0.0 if gamma is not provided

    print(f"Alpha value: {alpha}, Gamma value: {gamma}")  # Debugging print

    criterion = vpuLoss(gamma=gamma, alpha=alpha).to(device)

    for name, param in model.resnet.named_parameters():  # Change here to self.resnet
        param.requires_grad = "global_branch" in name

    # Accessing layer4 directly via resnet
    for param in model.resnet.layer4.parameters():  # Change here to self.resnet.layer4
        param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    train_multi_label_coco(model, train_loader, val_loader, args, sampler, criterion, scheduler, steps_per_epoch, optimizer, Epochs)

def train_multi_label_coco(model, train_loader, val_loader, args, sampler, criterion, scheduler, steps_per_epoch, optimizer, Epochs):
    ema = ModelEma(model, args.ema)
    Stop_epoch = args.Stop_epoch
    model.train()
    model.apply(fix_bn)

    for epoch in range(Epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if Stop_epoch is not None and epoch > Stop_epoch:
            break

        running_loss = 0.0
        num_batches = 0

        for i, batch in enumerate(train_loader):
            if batch is None or any(b is None for b in batch):
                continue

            sampleIndex, inputData, target, groundTruth = batch
            inputData = inputData.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(input=inputData)

            # Check if loss or output has NaN or Inf values
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"[DEBUG] NaN or Inf detected in output at epoch {epoch}, step {i}")
                continue

            loss = criterion(output, target)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"[DEBUG] NaN or Inf detected in loss at epoch {epoch}, step {i}")
                continue

            loss.backward()
            # Gradient clipping to avoid instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            num_batches += 1

            if i % 100 == 0:
                print(f'Epoch [{epoch}/{Epochs}], Step [{i}/{steps_per_epoch}], LR {scheduler.get_last_lr()[0]:.1e}, Loss: {loss.item():.4f}')

        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0.0
        print(f"\u2705 [Epoch {epoch}] Average Train Loss: {avg_train_loss:.4f}")

        model.eval()
        validate_multi(val_loader, model, ema, criterion)
        model.train()
        model.apply(fix_bn)

def validate_multi(val_loader, model, ema, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if batch is None or any(b is None for b in batch):
                continue

            sampleIndex, inputData, target, groundTruth = batch
            inputData = inputData.to(device)
            target = target.to(device)
            groundTruth = groundTruth.to(device)

            print(f"[DEBUG] groundTruth min: {groundTruth.min().item()}, max: {groundTruth.max().item()}")
            mask = groundTruth > 0
            print(f"[DEBUG] Valid samples in batch: {torch.sum(mask).item()}")

            if torch.sum(mask) == 0:
                print("[DEBUG] Skipping batch - No valid mask")
                continue

            output = model(input=inputData)

            # Check if loss or output has NaN or Inf values
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"[DEBUG] NaN or Inf detected in output at validation step {i}")
                continue

            loss = criterion(output[mask], groundTruth[mask])
            total_loss += loss.item()

            predicted = (output > 0.5).float()
            correct += (predicted[mask] == groundTruth[mask]).sum().item()
            total += mask.sum().item()

    avg_loss = total_loss / max(len(val_loader), 1)

    if total > 0:
        accuracy = correct / total * 100
        print(f"\u2705 Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    else:
        print(f"\u26A0\uFE0F Validation Loss: {avg_loss:.4f}, Accuracy: No valid samples processed.")

if __name__ == '__main__':
    main() 