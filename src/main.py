from dataset import get_data
from config import IMDB_URL, IMDB_FILE, UTILS_DIR
from modeling.models import SimpleSelfAttentionModelV1
from utils.data_rnn import load_imdb
from utils.helper_functions import read_review, create_batches
import torch


if __name__ == "main":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

    VOCAB_SIZE = len(i2w)
    EMB_SIZE = 300

    model_0 = SimpleSelfAttentionModelV1(
        vocab_size=VOCAB_SIZE, emb_size=EMB_SIZE, output_shape=numcls
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_0.parameters(), lr=0.001)
