from src.imports.common_imports import *
"""
The BilingualDataset class prepares bilingual translation data for use with transformer models. 
It handles tokenization, sequence padding, and mask generation. 
"""


class BilingualDataset(Dataset):
    def __init__(self, 
                 ds, 
                 src_lang_tokenizer, 
                 tgt_lang_tokenizer, 
                 src_lang: str, 
                 tgt_lang: str, 
                 seq_len: int
                 ) -> None:
        """
        Initialize the BilingualDataset module.

        Args:
            ds (Dataset): The original dataset containing source and target translations.
            src_lang_tokenizer: Tokenizer for the source language.
            tgt_lang_tokenizer: Tokenizer for the target language.
            src_lang (str): Source language identifier.
            tgt_lang (str): Target language identifier.
            seq_len (int): Maximum sequence length for input and output sequences.
        """
        super().__init__()
        self.seq_len = seq_len  # Store the maximum sequence length

        self.ds = ds  # Store the original dataset
        self.src_lang_tokenizer = src_lang_tokenizer  # Store the source language tokenizer
        self.tgt_lang_tokenizer = tgt_lang_tokenizer  # Store the target language tokenizer
        self.src_lang = src_lang  # Store the source language identifier
        self.tgt_lang = tgt_lang  # Store the target language identifier

        # Retrieve and store special tokens for the target language
        self.sos_token = torch.tensor([tgt_lang_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)  # Start-of-sequence token
        self.eos_token = torch.tensor([tgt_lang_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)  # End-of-sequence token
        self.pad_token = torch.tensor([tgt_lang_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)  # Padding token

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.ds)



    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the encoder input, decoder input, encoder mask,
                  decoder mask, label, source text, and target text.

                encoder_input (torch.Tensor): 
                A tensor of shape (seq_len,) representing the tokenized and padded source text, including 
                special tokens (<sos> and <eos>).

                decoder_input (torch.Tensor): 
                A tensor of shape (seq_len,) representing the tokenized and padded target text, including 
                the <sos> token and padding.
                
                encoder_mask (torch.Tensor): 
                A tensor of shape (1, 1, seq_len) representing the mask for the encoder input, where padding 
                tokens are masked out.

                decoder_mask (torch.Tensor):
                A tensor of shape (1, seq_len, seq_len) representing the mask for the decoder input, combining 
                padding masks and a causal mask to prevent attending to future tokens.

                label (torch.Tensor): 
                A tensor of shape (seq_len,) representing the tokenized and padded target text, including the 
                <eos> token and padding, which is used as the target output during training.

                src_text (str): 
                The original source text from the dataset.

                tgt_text (str): 
                The original target text from the dataset.
        """
        # Retrieve the source-target pair from the dataset
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]  # Extract source text
        tgt_text = src_target_pair['translation'][self.tgt_lang]  # Extract target text

        # Transform the text into tokens using the respective tokenizers
        enc_input_tokens = self.src_lang_tokenizer.encode(src_text).ids
        dec_input_tokens = self.tgt_lang_tokenizer.encode(tgt_text).ids

        # Calculate the number of padding tokens needed for both encoder and decoder inputs
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # For encoder: account for <s> and </s>
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # For decoder: account for <s>

        # Ensure the sequences are not too long for the specified maximum length
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Create the encoder input by adding <s>, </s>, and padding tokens
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Create the decoder input by adding <s> and padding tokens
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Create the label by adding </s> and padding tokens
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Ensure the size of the tensors matches the specified sequence length
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # Create the encoder mask to ignore padding tokens
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()  # Shape: (1, 1, seq_len)

        # Create the decoder mask to ignore padding tokens and prevent attending to future tokens
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0))  # Shape: (1, seq_len) & (1, seq_len, seq_len)

        # Return the processed input, mask, and original texts as a dictionary
        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": encoder_mask,  # (1, 1, seq_len)
            "decoder_mask": decoder_mask,  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size: int) -> torch.Tensor:
    """
    Create a causal mask for the decoder to prevent attending to future tokens.

    Args:
        size (int): The size of the sequence.

    Returns:
        torch.Tensor: A causal mask tensor of shape (1, size, size) with the upper triangular part masked.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
