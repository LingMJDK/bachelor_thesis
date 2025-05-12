import torch

def generate_text(
    model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, device="cpu"
):
    """
    Generate text from an autoregressive transformer model.
    Params:
        model: Autoregressive Transformer model (trained)
        tokenizer: Tokenizer with tensor_encoding and tensor_decoding
        prompt: Starting string
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (>1 = more random, <1 = more confident)
        device: "cuda" or "cpu"
    Returns:
        Generated string including prompt
    """

    model.eval()
    model.to(device)

    # Encode the prompt to input tensor
    input_ids = (
        tokenizer.tensor_encoding(prompt).unsqueeze(0).to(device)
    )  # shape: (1, T)

    for _ in range(max_new_tokens):
        # Forward pass
        with torch.inference_mode():
            y_logits = model(input_ids)  # shape: (1, T, vocab_size)

        # Select logits of last token in sequence
        next_token_logits = y_logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)

        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)

        # Append prediction to input
        input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode the full sequence (including the prompt)
    generated_text = tokenizer.tensor_decoding(input_ids[0])
    return generated_text
