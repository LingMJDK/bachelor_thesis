import argparse
import os
import torch
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer as Timer
import wandb


try:
    from src.dataset import load_wiki_dataset, char_batches
    from modeling.models import AutoregressiveTransformer
    from modeling.train import train_step, accuracy_fnV2, test_step
    from modeling.predict import generate_text
except ImportError:
    from dataset import load_wiki_dataset, char_batches
    from modeling.models import AutoregressiveTransformer
    from modeling.train import train_step, accuracy_fnV2, test_step
    from modeling.predict import generate_text

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an autoregressive transformer on the Wiki dataset"
    )
    parser.add_argument("--num-train",       type=int,   default=100,   help="Number of training batches")
    parser.add_argument("--epochs",          type=int,   default=5,     help="Number of epochs")
    parser.add_argument("--batch-size",      type=int,   default=10,     help="Batch size")
    parser.add_argument("--seq-len",         type=int,   default=110,   help="Sequence length")
    parser.add_argument("--seed",            type=int,   default=22,    help="Random seed")
    parser.add_argument("--emb-size",        type=int,   default=256,   help="Embedding dimension size")
    parser.add_argument("--n-layers",        type=int,   default=4,     help="Number of transformer layers")
    parser.add_argument("--n-heads",         type=int,   default=4,     help="Number of attention heads")
    parser.add_argument("--ff-hidden-mult",  type=int,   default=4,     help="Feedforward hidden size multiplier")
    parser.add_argument("--dropout",         type=float, default=0.1,   help="Dropout probability")
    parser.add_argument("--lr",              type=float, default=1e-4,  help="Learning rate for optimizer")
    parser.add_argument("--output-dir",      type=str,   default="results", help="Directory for checkpoints and logs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # ─── reproducibility & dirs ──────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── wandb init & config ─────────────────────────────────────────────
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
        dir=args.output_dir,    # write wandb files into your results folder
    )
    config = wandb.config

    # ─── device & data ───────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    train_data, val_data, test_data, tokenizer = load_wiki_dataset()

    # Prepare iterators
    num_val  = int(0.2 * config.num_train)
    num_test = num_val
    train_iter = char_batches(train_data, config.seq_len, config.batch_size,
                              config.num_train, tokenizer.tensor_encoding,
                              seed=config.seed, replacement=True)
    val_iter   = char_batches(val_data,   config.seq_len, config.batch_size,
                              num_val,        tokenizer.tensor_encoding,
                              seed=config.seed, replacement=True)
    test_iter  = char_batches(test_data,  config.seq_len, config.batch_size,
                              num_test,       tokenizer.tensor_encoding,
                              seed=config.seed, replacement=True)

    train_set = list(train_iter)
    val_set   = list(val_iter)
    test_set  = list(test_iter)

    print(f"Train batches: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # ─── model, loss, optimizer ─────────────────────────────────────────
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

    # Watch for gradients / parameter histograms
    wandb.watch(model, log="all", log_freq=10)

    # ─── training loop ───────────────────────────────────────────────────
    for epoch in range(config.epochs):
        start = Timer()

        train_loss, train_acc = train_step(
            model=model, train_data=train_set,
            loss_fn=loss_fn, accuracy_fn=accuracy_fn,
            optimizer=optimizer, epoch=epoch,
            device=device, task="Autoregressive"
        )

        val_loss, val_acc = test_step(
            model=model, test_data=val_set,
            loss_fn=loss_fn, accuracy_fn=accuracy_fn,
            epoch=epoch, device=device, task="Autoregressive"
        )

        elapsed = Timer() - start
        print(f"Epoch {epoch} in {elapsed:.2f}s — "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # ─── log to wandb ──────────────────
        wandb.log({
            "epoch":            epoch,
            "train/loss":       train_loss,
            "train/accuracy":   train_acc,
            "val/loss":         val_loss,
            "val/accuracy":     val_acc,
            "lr":               optimizer.param_groups[0]['lr'],
            "time/epoch_sec":   elapsed,
        })

        # ─── checkpoint ───────────
        if epoch % 50 == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "opt_state":   optimizer.state_dict(),
                "config":      dict(config),
            }, ckpt_path)
            wandb.save(ckpt_path)   # upload to W&B run

    wandb.finish()
    print("Done.")


