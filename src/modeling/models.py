import torch
from torch import nn


class GlobalMeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Extracts the mean across the time (sequence) dimension and returns the values of the embedded vector."""
        return torch.mean(x, dim=1)


class GlobalMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Extracts the max across the time (sequence) dimension and returns the values of the embedded vector."""
        return torch.max(x, dim=1)[0]


class SentimentClassificationModelV0(nn.Module):
    def __init__(self, vocab_size, emb_size, output_shape):
        """
        Params
          vocab_size: The size of the vocabulary.
          emb_size: The size of the embedding vector
          output_shape:
        """
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Embedding(
                num_embeddings=vocab_size,  # Convert the integer indices to embedding vectors
                embedding_dim=emb_size,
                padding_idx=0,
            ),
            GlobalMaxPooling(),  # Global max pooling along sequence dimension
            # nn.ReLU(),
            nn.Linear(
                in_features=emb_size,  # Project down to the number of classes
                out_features=output_shape,
            ),
        )

    def forward(self, x):
        return self.sequence(x)


class SentimentClassificationModelV1(nn.Module):
    def __init__(self, vocab_size, emb_size, output_shape):
        """
        Params
          vocab_size: The size of the vocabulary.
          emb_size: The size of the embedding vector
          output_shape:
        """
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Embedding(
                num_embeddings=vocab_size,  # Convert the integer indices to embedding vectors
                embedding_dim=emb_size,
                padding_idx=0,
            ),
            GlobalMeanPooling(),  # Global max pooling along sequence dimension
            # nn.ReLU(),
            nn.Linear(
                in_features=emb_size,  # Project down to the number of classes
                out_features=output_shape,
            ),
        )

    def forward(self, x):
        return self.sequence(x)


class SimpleSelfAttentionModelV0(nn.Module):
    def __init__(
        self, vocab_size, emb_size: int, output_shape: int, padding_idx: int = 0
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb_size, padding_idx=padding_idx
        )

        self.linear_layer = nn.Linear(in_features=emb_size, out_features=output_shape)

    def forward(self, x: torch.tensor):
        embeddings = self.embedding(x)  # (batch_size, seq_length, emb_size)
        attention_weight_scores = torch.bmm(embeddings, embeddings.transpose(1, 2))
        attention_weigths = torch.softmax(attention_weight_scores, dim=2)
        output_vector = torch.bmm(attention_weigths, embeddings)
        max_pool = torch.max(output_vector, dim=1)[0]

        output = self.linear_layer(max_pool)
        return output
    

class MultiHeadSelfAttentionModelV1(nn.Module):
    def __init__(self, vocab_size, emb_size, output_shape, n_heads=4, padding_idx=0):
        super().__init__()

        self.n_heads = n_heads
        self.k = emb_size // n_heads

        assert emb_size % n_heads == 0, "Embedding size must be a multiple of the number of heads"

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)

        # linear projection for Q, K, V (all heads combined!)
        self.W_q = nn.Linear(emb_size, emb_size)
        self.W_k = nn.Linear(emb_size, emb_size)
        self.W_v = nn.Linear(emb_size, emb_size)

        # Final projection after heads are concatenated
        self.output_proj = nn.Linear(emb_size, emb_size)

        # Final output classifier layer
        self.classifier = nn.Linear(emb_size, output_shape)


    def forward(self, x):
        batch_size, seq_len = x.shape
        embeddings = self.embedding(x)  # (batch, seq, emb)

        Q = self.W_q(embeddings)  # (batch, seq, emb)
        K = self.W_k(embeddings)
        V = self.W_v(embeddings)

        # Split heads: (batch, seq, emb) → (batch, n_heads, seq, k)
        split_heads = lambda x: x.view(batch_size, seq_len, self.n_heads, self.k).transpose(1,2)

        Q = split_heads(Q).contiguous()
        K = split_heads(K).contiguous()
        V = split_heads(V).contiguous()

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.k ** 0.5)  # (batch, n_heads, seq, seq)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)  # (batch, n_heads, seq, k)

        # Concatenate heads (batch, n_heads, seq, d_k) ---> (batch, seq, emb)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Final projection
        output_vector = self.output_proj(attention_output)  # (batch, seq, emb)

        max_pooled = torch.max(output_vector, dim=1)[0]  # (batch, emb)

        return self.classifier(max_pooled)
    