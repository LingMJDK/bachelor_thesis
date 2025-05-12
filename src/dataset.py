import os
import requests
from config import ENWIK8_PATH
from utils.util import enwik8_string
import random
import textwrap
from typing import List, Tuple, Generator, Dict
import torch
from data_rnn import i2w


def create_batches(data: List[List[int]], labels: List[int]) -> List[List[List[int]]]:
    """
    Takes list of reviews and labels and turns them into batches of labels and reviews of uniform length.
    Params
      data: A list of reviews.
      labels: A list of labels
    Returns
      A list of batches containing reviews, each padded to ensure uniform length.
    """
    data, labels = data.copy(), labels.copy()
    data_and_labels = zip(data, labels)

    # Sorts the data from longest to shortest reviews and unzips data from labels
    data, labels = list(
        zip(*sorted(data_and_labels, key=lambda x: len(x[0]), reverse=True))
    )

    data_batch: List[List[int]] = []
    label_batch: List[int] = []

    data_batches: List[List[List[int]]] = []
    label_batches: List[List[int]] = []

    curr_review_length: int = 0

    # Determines the size of the longest review
    max_tokens: int = max(len(review) for review in data)

    # loop over the list of reviews (x_train)
    for review_idx in range(len(data)):
        curr_review: List[int] = data[review_idx]
        curr_label: int = labels[review_idx]
        num_tokens: int = sum(
            len(review) for review in data_batch
        )  # Calculate the number of tokens in the data_batch

        # If data_batch is empty add a review and set current review length
        if num_tokens == 0:
            data_batch.append(curr_review)
            label_batch.append(curr_label)
            curr_review_length = len(curr_review)

        # If the number of tokens in a data_batch exceed the limit or adding an extra token of the same size exceeds the limit:
        elif (num_tokens >= max_tokens) or (
            num_tokens + curr_review_length >= max_tokens
        ):
            # Add the current data_batch to the list of data_batches
            data_batches.append(data_batch)
            label_batches.append(label_batch)
            # Create an empty data_batch for the next reviews
            data_batch = []
            label_batch = []
            # Append current review to new data_batch
            data_batch.append(curr_review)
            label_batch.append(curr_label)
            # Reset current review length
            curr_review_length = len(curr_review)

        else:
            padding = curr_review_length - len(
                curr_review
            )  # Calculate padding needed for next token
            curr_review += padding * [0]  # Add padding to next token
            data_batch.append(curr_review)  # Add a review to the data_batch list
            label_batch.append(curr_label)

    # Ensures that the final data_batch is also added.
    if data_batch:
        data_batches.append(data_batch)
        label_batches.append(label_batch)

    return data_batches, label_batches


#################################################################################


def read_review(review: List[int]) -> None:
    """
    Takes a list of integers as input and prints their string representation.
    Params:
      review: A list of integers.
    Returns:
      None
    """
    print(
        textwrap.fill(" ".join([i2w[word] for word in review]), 100),
    )


##################################################################################


def show_batches(
    data_batches, label_batches, batch_num: int = 0, print_reviews: bool = True
) -> None:
    """
    Filler.
    """
    class_name = ["Positive", "Negative"]

    num_data_batches = len(data_batches)
    num_label_batches = len(label_batches)

    data_batch = data_batches[batch_num]
    label_batch = label_batches[batch_num]

    num_tokens = sum(len(review) for review in data_batch)

    review_length = len(data_batch[0])

    # check if reviews are all the same size
    same_size = max(len(review) for review in data_batch) == min(
        len(review) for review in data_batch
    )

    print(
        f"Number of data batches: {num_data_batches} | Number of label batches {num_label_batches}",
        f"\nBatch number: {batch_num}\n____________________________________________\n",
    )
    print(f"Number of tokens in batch: {num_tokens}")
    print(
        f"Review length: {review_length}", f"\nReviews are the same size: {same_size}"
    )
    print(
        f"Data batch size: {len(data_batch)} | Label batch size: {len(label_batch)}",
        "\n_____________________________________________\n",
    )

    if print_reviews:
        for review, label in zip(data_batch, label_batch):
            read_review(review)
            print(
                f"\nLabel: {class_name[label]}",
                "\n------------------------------------",
            )


#############################################################################################


def data_to_tensor_batch(
    data: List[list[int]], labels: List[int], device: str
) -> Tuple[Generator[torch.Tensor, None, None], Generator[torch.Tensor, None, None]]:
    """
    Filler.
    """
    x_batches, y_batches = create_batches(data, labels)
    x_batched_tensors = (
        torch.tensor(batch, dtype=torch.long, device=device) for batch in x_batches
    )
    y_batched_tensors = (
        torch.tensor(batch, dtype=torch.long, device=device) for batch in y_batches
    )

    return x_batched_tensors, y_batched_tensors


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time."""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def char_batches(
    data, seq_len, batch_size, num_batches, encoding, seed=22, replacement=True
):
    """
    Params:
      data: A string of text.
      seq_len: The sequence length for each instance in the batch.
      batch_size: The size of each batch.
      num_batches: The number of batches.
      encoding:
      replacement:
    Returns
      .
    """
    random.seed(seed)
    batches = []

    # The max slicing index
    max_index = len(data) - (seq_len + 1)

    # Assert if batch_size * batches does not exceed number of unique instances
    assert num_batches * batch_size < max_index, (
        "The number of (batch_size * batches) exceeds the number of unique instances"
    )

    if replacement:
        index_gen = iter(
            [random.randint(0, max_index) for _ in range(num_batches * batch_size)]
        )
    else:
        index_gen = iter(random.sample(range(max_index), k=num_batches * batch_size))

    for _ in range(num_batches):
        Xy = [
            (encoding(data[i : i + seq_len]), encoding(data[i + 1 : i + (seq_len + 1)]))
            for i in [next(index_gen) for _ in range(batch_size)]
        ]
        # Unzip X and y
        X, y = zip(*Xy)

        # Stack tensors to create single tensor
        batches.append((torch.stack(X), torch.stack(y)))

    return iter(batches)


class Tokenizer:
    def __init__(self, data: str):
        """ """
        self.chars = sorted(list(set(data)))
        self.ch_to_i: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.i_to_ch: Dict[int, str] = {i: ch for ch, i in self.ch_to_i.items()}
        self.vocab_size: int = len(self.chars)
        self.data = data

    def __repr__(self):
        """ """

        return f"Vocabulary of size: {self.vocab_size}"

    def list_encoding(self, string):
        """ """

        return [self.ch_to_i[char] for char in string]

    def tensor_encoding(self, string):
        """ """

        return torch.tensor([self.ch_to_i[char] for char in string], dtype=torch.long)

    def list_decoding(self, char_list):
        """ """

        return "".join([self.i_to_ch[i] for i in char_list])

    def tensor_decoding(self, char_tensor: torch.tensor):
        """ """

        return "".join([self.i_to_ch[i] for i in char_tensor.tolist()])

    def vocab_ch_to_i(self):
        """ """

        return self.ch_to_i

    def vocab_i_to_ch(self):
        """ """

        return self.i_to_ch

    def get_vocab_size(self):
        """ """

        return self.vocab_size

    def get_data(self):
        """ """
        return self.data

    def get_data_size(self):
        """ """
        return len(self.data)


def get_data(data_dir: str, url: str, filename: str) -> str:
    """
    Downloads the data from the given URL if it is not already present in the data directory.
    Returns the path to the data file.
    Params:
    - data_dir: The directory where the data file will be stored.
    Returns:
    - The path to the data file.
    """
    # Create the data directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    # Path to the data file
    data_path = os.path.join(data_dir, filename)

    # Download the file if it does not exist
    if not os.path.exists(data_path):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(data_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {filename} to {data_path}")
    else:
        print(f"{filename} already exists. Skipping download.")


def load_wiki_dataset():
    # Load the enwik8 dataset
    train_data, val_data, test_data = enwik8_string(str(ENWIK8_PATH))
    return train_data, val_data, test_data


def tokenize_data(full_data_set):
    pass
