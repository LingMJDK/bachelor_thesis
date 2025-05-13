import argparse
import os
import torch
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer as Timer
import wandb

from torch.utils.data import DataLoader
from dataset import CharBatchDataset, load_wiki_dataset
from modeling.models import AutoregressiveTransformer
from modeling.train import train_step, accuracy_fnV2, test_step
from modeling.predict import generate_text

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an autoregressive transformer on the Wiki dataset"
    )
    parser.add_argument("--num-train",       type=int,   default=10,   help="Number of training batches")
    parser.add_argument("--epochs",          type=int,   default=5,     help="Number of epochs")
    parser.add_argument("--batch-size",      type=int,   default=2,     help="Batch size")
    parser.add_argument("--seq-len",         type=int,   default=256,   help="Sequence length")
    parser.add_argument("--seed",            type=int,   default=22,    help="Random seed")
    parser.add_argument("--emb-size",        type=int,   default=512,   help="Embedding dimension size")
    parser.add_argument("--n-layers",        type=int,   default=6,     help="Number of transformer layers")
    parser.add_argument("--n-heads",         type=int,   default=8,     help="Number of attention heads")
    parser.add_argument("--ff-hidden-mult",  type=int,   default=4,     help="Feedforward hidden size multiplier")
    parser.add_argument("--dropout",         type=float, default=0.1,   help="Dropout probability")
    parser.add_argument("--lr",              type=float, default=1e-4,  help="Learning rate for optimizer")
    parser.add_argument("--output-dir",      type=str,   default="results", help="Directory for checkpoints and logs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # reproducibility & dirs
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # device setup & cuDNN tuning
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # wandb init & config
    wandb.init(
        project="wiki-transformer",
        config={
            "num_train":    args.num_train,
            "epochs":       args.epochs,
            "batch_size":   args.batch_size,
            "seq_len":      args.seq_len,
            "seed":         args.seed,
            "emb_size":     args.emb_size,
            "n_layers":     args.n_layers,
            "n_heads":      args.n_heads,
            "ff_hidden_mult": args.ff_hidden_mult,
            "dropout":      args.dropout,
            "lr":           args.lr,
        },
        dir=args.output_dir,
    )
    config = wandb.config

    # data loading via IterableDataset + DataLoader
    train_data, val_data, test_data, tokenizer = load_wiki_dataset()
    num_val  = int(0.2 * config.num_train)
    num_test = num_val

    train_ds = CharBatchDataset(
        data=train_data,
        encoding_fn=tokenizer.tensor_encoding,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        num_batches=config.num_train,
        seed=config.seed,
        replacement=True,
    )
    val_ds = CharBatchDataset(
        data=val_data,
        encoding_fn=tokenizer.tensor_encoding,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        num_batches=num_val,
        seed=config.seed,
        replacement=True,
    )
    test_ds = CharBatchDataset(
        data=test_data,
        encoding_fn=tokenizer.tensor_encoding,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        num_batches=num_test,
        seed=config.seed,
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=None,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=None,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    print(f"Train batches: {config.num_train}, Val: {num_val}, Test: {num_test}")

    # model, loss, optimizer
    model = AutoregressiveTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        emb_size=config.emb_size,
        max_seq_len=config.seq_len,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        ff_hidden_mult=config.ff_hidden_mult,
        dropout=config.dropout,
        padding=False
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    accuracy_fn = accuracy_fnV2

    wandb.watch(model, log="all", log_freq=10)

    # training loop
    for epoch in range(config.epochs):
        start = Timer()

        train_loss, train_acc = train_step(
            model=model,
            train_data=train_loader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            task="Autoregressive"
        )

        val_loss, val_acc = test_step(
            model=model,
            test_data=val_loader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
            epoch=epoch,
            task="Autoregressive"
        )

        elapsed = Timer() - start
        print(f"Epoch {epoch} in {elapsed:.2f}s — "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        wandb.log({
            "epoch":            epoch,
            "train/loss":       train_loss,
            "train/accuracy":   train_acc,
            "val/loss":         val_loss,
            "val/accuracy":     val_acc,
            "lr":               optimizer.param_groups[0]['lr'],
            "time/epoch_sec":   elapsed,
        })
        if epoch % 4 == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "opt_state":   optimizer.state_dict(),
                "config":      dict(config),
            }, ckpt_path)
            wandb.save(ckpt_path)

    wandb.finish()
    print("Done.")
