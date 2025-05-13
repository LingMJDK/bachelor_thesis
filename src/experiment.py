import argparse
import os
import torch
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer as Timer
import wandb
import argparse
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import random
import torch
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer as Timer
import wandb
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.utils.data import DataLoader
from src.dataset import CharBatchDataset, load_wiki_dataset
from modeling.models import AutoregressiveTransformer
from modeling.train import train_step, accuracy_fnV2, test_step
from modeling.predict import generate_text


def parse_args():
    parser = argparse.ArgumentParser(description="Train an autoregressive transformer on the Wiki dataset")
    parser.add_argument("--num-train", type=int, default=100)
    parser.add_argument("--num-val", type=int, default=500)
    parser.add_argument("--num-test", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=110)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--emb-size", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-hidden-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda": torch.backends.cudnn.benchmark = True

    wandb.init(project="wiki-transformer", config=vars(args), dir=args.output_dir)
    config = wandb.config

    # Import the original loader
    from src.dataset import load_wiki_dataset
    from modeling.models import AutoregressiveTransformer

    train_data, val_data, test_data, tokenizer = load_wiki_dataset()
    num_test = config.num_test
    num_val = config.num_val

    # Create DataLoaders
    train_ds = CharBatchDataset(train_data, tokenizer.tensor_encoding,
                                config.seq_len, config.batch_size,
                                config.num_train, seed=config.seed)
    val_ds   = CharBatchDataset(val_data,   tokenizer.tensor_encoding,
                                config.seq_len, config.batch_size,
                                num_val, seed=config.seed)
    train_loader = DataLoader(train_ds, batch_size=None, num_workers=4,
                              pin_memory=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=None, num_workers=2,
                              pin_memory=True, prefetch_factor=2)

    model = AutoregressiveTransformer(
        vocab_size=tokenizer.get_vocab_size(), emb_size=config.emb_size,
        max_seq_len=config.seq_len, n_layers=config.n_layers,
        n_heads=config.n_heads, ff_hidden_mult=config.ff_hidden_mult,
        dropout=config.dropout, padding=False
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    accuracy_fn = globals()['accuracy_fnV2'] if 'accuracy_fnV2' in globals() else None

    wandb.watch(model, log="all", log_freq=10)

    for epoch in range(config.epochs):
        train_ds.seed = config.seed + epoch
        start = Timer()
        train_loss, train_acc = train_step(model, train_loader, loss_fn,
                                           accuracy_fn, optimizer, device,
                                           epoch, task="autoregressive")
        val_loss, val_acc = test_step(model, val_loader, loss_fn,
                                      accuracy_fn, device, epoch,
                                      task="autoregressive")
        elapsed = Timer() - start
        print(f"Epoch {epoch} in {elapsed:.2f}s —"
              f" train_loss={train_loss:.4f}, train_acc={train_acc:.4f},"
              f" val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        wandb.log({"epoch": epoch,
                   "train/loss": train_loss,
                   "train/accuracy": train_acc,
                   "val/loss": val_loss,
                   "val/accuracy": val_acc,
                   "lr": optimizer.param_groups[0]['lr'],
                   "time/epoch_sec": elapsed})
        if epoch % 50 == 0:
            ckpt = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "opt_state": optimizer.state_dict(),
                        "config": vars(config)}, ckpt)
            wandb.save(ckpt)

    wandb.finish()
    print("Done.")
