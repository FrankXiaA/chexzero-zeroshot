import os
import pprint
import argparse
from tqdm import tqdm

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize

import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer

from train import train_main, load_data, load_clip, preprocess_text
from zero_shot import run_cxr_zero_shot, run_zero_shot

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='data/cxr.h5',
                        help="Directory to load chest x-ray image data from.")
    parser.add_argument('--txt_filepath', type=str, default='data/mimic_impressions.csv',
                        help="Directory to load radiology report impressions text from.")
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to the pretrained model weights (None for random weights).")
    parser.add_argument('--lora_r', type=int, default=8, help="Dimension of low-rank approximation for LoRA.")
    parser.add_argument('--pretrained', action='store_false', help="Use pretrained weights (default: random weights).")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training (default: 64).")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs (default: 50).")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate (default: 1e-3 for SGD).")
    parser.add_argument('--save_interval', type=int, default=10,
                        help="Save model checkpoint every N epochs (default: 10).")
    parser.add_argument('--log_interval', type=int, default=10,
                        help="Log training metrics every N batches (default: 10).")
    parser.add_argument('--save_dir', type=str, default="checkpoints/", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed for reproducibility (default: 1234).")
    parser.add_argument('--optimizer', type=str, default="adam", help="Optimizer to use: sgd or adam (default: adam).")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum for SGD (default: 0.9).")
    parser.add_argument('--context_length', type=int, default=77,
                        help="Maximum token length for text input (default: 77).")
    parser.add_argument('--random_init', action='store_true', default=True,
                        help="Initialize model weights randomly (default: True).")
    parser.add_argument('--model_name', type=str, default="few-shot",
                        help="Name for the model being trained (default: 'few-shot').")

    args = parser.parse_args()
    return args


def model_pipeline(config, verbose=0):
    """
    The main pipeline for model training. Handles model initialization, data loading, and training.
    """
    # Handle random initialization
    if config.random_init:
        print("Training with randomly initialized weights.")
        model_path = None
        pretrained = False
    else:
        print(f"Using pretrained weights from {config.model_path}")
        model_path = config.model_path
        pretrained = True

    # Make the model, data, and optimization problem
    model, data_loader, device, criterion, optimizer = make(config,model_path=model_path,pretrained=pretrained)

    # Train the model
    train(model, data_loader, device, optimizer, config)

    # Save the model
    model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint.pt')
    save(model, model_path)

    if verbose:
        print(model)
    return model


def make(config, model_path=None, pretrained=False):
    """
    Creates the model, data loader, loss function, and optimizer.
    """
    data_loader, device = load_data(
        config.cxr_filepath,
        config.txt_filepath,
        batch_size=config.batch_size,
        pretrained=pretrained,
        column="impression"
    )

    model = load_clip(
        model_path=model_path,
        pretrained=pretrained,
        context_length=config.context_length,
        lora_r=config.lora_r
    )
    model.to(device)
    print('Model initialized and moved to device.')

    # Create the optimizer and loss function
    criterion = nn.CrossEntropyLoss().cuda()
    if config.optimizer == "adam":
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    return model, data_loader, device, criterion, optimizer



def train(model, loader, device, optimizer, config):
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    if not os.path.exists(model_save_dir):
        # Create a new folder if not exists
        os.makedirs(model_save_dir)

    # Run training
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    report_freq = config.log_interval
    highest_val_auc = 0  # save highest mean auc

    for epoch in range(config.epochs):
        running_loss = 0.0  # running loss over batch
        for data in tqdm(loader):
            # Get the images and texts
            images = data['img']
            texts = data['txt']
            texts = preprocess_text(texts, model)

            # Perform step for a single batch
            loss = train_batch(images, texts, model, device, optimizer)
            example_ct += len(images)
            batch_ct += 1
            running_loss += loss.item()

            # Report metrics every `report_freq` batch
            if (batch_ct % report_freq) == 0:
                train_log(running_loss / report_freq, example_ct, epoch)
                running_loss = 0.0

            # Save checkpoint at intervals defined by `config.save_interval`
            if (batch_ct % config.save_interval) == 0:
                model_path = os.path.join(model_save_dir, f"checkpoint_{batch_ct}.pt")
                print("Saved checkpoint to: ", model_path)
                save(model, model_path)

import torch
import torch.nn.functional as F


def train_batch(images, texts, model, device, optimizer):
    """TODO: Compute loss based on cosine similarity as logits."""
    images, texts = images.to(device), texts.to(device)

    # Forward pass ➡
    image_embeddings, text_embeddings = model(images, texts)

    # Normalize the embeddings to ensure cosine similarity is between -1 and 1
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    # Compute cosine similarity
    logits_per_image = image_embeddings @ text_embeddings.T
    logits_per_text = text_embeddings @ image_embeddings.T

    # Create labels
    batch_size = images.shape[0]
    labels = torch.arange(batch_size).to(device)

    # Compute loss using cross-entropy on the cosine similarity logits
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)
    loss = (loss_img + loss_txt) / 2  # average the image and text loss

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    
def save(model, path): 
    torch.save(model.state_dict(), path)
    
if __name__ == "__main__":
    args = parse_args()
    model = model_pipeline(args)


