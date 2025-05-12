import argparse
import os
import torch
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer as Timer

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
    parser.add_argument("--epochs",          type=int,   default=1,     help="Number of epochs")
    parser.add_argument("--batch-size",      type=int,   default=64,    help="Batch size")
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
    os.makedirs(args.output_dir, exist_ok=True)

    # Device-agnostic setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Hyperparameters
    NUM_TRAIN        = args.num_train
    EPOCHS           = args.epochs
    NUM_VAL          = int(0.2 * NUM_TRAIN)
    NUM_TEST         = int(0.2 * NUM_TRAIN)
    BATCH_SIZE       = args.batch_size
    SEQ_LEN          = args.seq_len
    SEED             = args.seed
    EMB_SIZE         = args.emb_size
    N_LAYERS         = args.n_layers
    N_HEADS          = args.n_heads
    FF_HIDDEN_MULT   = args.ff_hidden_mult
    DROPOUT          = args.dropout
    LR               = args.lr

    # Load dataset
    train_data, val_data, test_data, tokenizer = load_wiki_dataset()
    full_data = train_data + val_data + test_data

    # Prepare data batches
    train_iter = char_batches(
        data=train_data,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        num_batches=NUM_TRAIN,
        encoding=tokenizer.tensor_encoding,
        seed=SEED,
        replacement=True
    )
    val_iter = char_batches(
        data=val_data,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        num_batches=NUM_VAL,
        encoding=tokenizer.tensor_encoding,
        seed=SEED,
        replacement=True
    )
    test_iter = char_batches(
        data=test_data,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        num_batches=NUM_TEST,
        encoding=tokenizer.tensor_encoding,
        seed=SEED,
        replacement=True
    )
    train_set, val_set, test_set = list(train_iter), list(val_iter), list(test_iter)

    # Basic sanity checks
    assert set(train_data).issubset(set(full_data))
    assert set(val_data).issubset(set(full_data))
    assert set(test_data).issubset(set(full_data))

    print(f"Train batches: {len(train_set)}, Val batches: {len(val_set)}, Test batches: {len(test_set)}")

    # Initialize model, loss and optimizer
    model = AutoregressiveTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        emb_size=EMB_SIZE,
        max_seq_len=SEQ_LEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ff_hidden_mult=FF_HIDDEN_MULT,
        dropout=DROPOUT,
        padding=False
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    accuracy_fn = accuracy_fnV2

    # Training loop with checkpointing
    for epoch in range(EPOCHS):
        epoch_start = Timer()

        # Training step
        train_step(
            model=model,
            train_data=train_set,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            task='Autoregressive'
        )

        # Validation step
        test_step(
            model=model,
            test_data=val_set,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            epoch=epoch,
            device=device,
            task='Autoregressive'
        )

        # Timing and logging
        epoch_end = Timer()
        elapsed = epoch_end - epoch_start
        print(f"Epoch {epoch} completed in {elapsed:.2f}s on {device}")

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": SEED,
            "hyperparameters": {
                "emb_size": EMB_SIZE,
                "n_layers": N_LAYERS,
                "n_heads": N_HEADS,
                "ff_hidden_mult": FF_HIDDEN_MULT,
                "dropout": DROPOUT,
                "learning_rate": LR
            }
        }, ckpt_path)
        print(f"Checkpoint saved to: {ckpt_path}")
