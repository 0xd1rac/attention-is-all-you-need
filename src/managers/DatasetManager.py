from src.imports.common_imports import *  # Importing common libraries and modules
import src.data as data  # Importing data module from src package

class DatasetManager:
    def __init__(self, 
                 tokenizer_file: str,  # Path template for saving/loading tokenizers
                 lang_src: str,  # Source language code
                 lang_tgt: str,  # Target language code
                 seq_len: int,  # Maximum sequence length for tokenization
                 batch_size: int  # Batch size for DataLoader
                ):
        self.tokenizer_file = tokenizer_file
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len
        self.batch_size = batch_size
    
    def get_all_sentences(self, ds, lang):
        """
        Generator function to yield sentences from the dataset for a specific language.
        """
        for item in ds:
            yield item['translation'][lang]
    
    def get_tokenizer(self, ds, lang):
        """
        Retrieves or trains a tokenizer for the specified language.

        Attention paper uses byte-pair encoding (BPE) as the tokenization method
        """
        tokenizer_path = Path(self.tokenizer_file.format(lang))
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not Path.exists(tokenizer_path):  # Check if tokenizer file already exists
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))  # Initialize tokenizer
            tokenizer.pre_tokenizer = Whitespace()  # Use whitespace for tokenization
            trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2) # A word-level tokenizer treats each word in the text as a token.
            tokenizer.train_from_iterator(self.get_all_sentences(ds, lang), trainer=trainer)  # Train tokenizer
            tokenizer.save(str(tokenizer_path))  # Save tokenizer to file
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))  # Load tokenizer from file

        return tokenizer

    def get_dataset(self, train_pct=0.9):
        """
        Loads and preprocesses the dataset, splits it into training and validation sets,
        and creates DataLoader instances.
        """
        lang_config = f"{self.lang_src}-{self.lang_tgt}"  # Language configuration
        ds_raw = load_dataset("opus_books", lang_config, split='train')  # Load dataset

        # Reduce dataset size for debugging and testing
        subset_size = len(ds_raw) // 80
        all_indices = list(range(len(ds_raw)))
        subset_indices = random.sample(all_indices, subset_size)
        ds_raw = Subset(ds_raw, subset_indices)

        # Build tokenizers for source and target languages
        tokenizer_src = self.get_tokenizer(ds_raw, self.lang_src)
        tokenizer_tgt = self.get_tokenizer(ds_raw, self.lang_tgt)

        # Split dataset into training and validation sets
        train_ds_size = int(train_pct * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

        # Create BilingualDataset instances for training and validation sets
        train_ds = data.BilingualDataset(train_ds_raw,
                                         tokenizer_src,
                                         tokenizer_tgt,
                                         self.lang_src,
                                         self.lang_tgt,
                                         self.seq_len)
        val_ds = data.BilingualDataset(val_ds_raw,
                                       tokenizer_src,
                                       tokenizer_tgt,
                                       self.lang_src,
                                       self.lang_tgt,
                                       self.seq_len)

        # Calculate maximum lengths of tokenized sentences in the dataset
        max_len_src = 0
        max_len_tgt = 0

        for item in ds_raw:
            src_ids = tokenizer_src.encode(item['translation'][self.lang_src]).ids
            tgt_ids = tokenizer_tgt.encode(item['translation'][self.lang_tgt]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f"Max length of source sentence: {max_len_src}")
        print(f"Max length of target sentence: {max_len_tgt}")

        # Create DataLoaders for training and validation datasets
        train_dataloader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

        return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
