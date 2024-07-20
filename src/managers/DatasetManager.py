from ManagerImports import *
from ConfigManager import ConfigManager
from imports import *
from BilingualDataset import BilingualDataset


class DatasetManager:
    def __init__(self, 
                tokenizer_file: str,
                lang_src: str,
                lang_tgt: str,
                seq_len:int,
                batch_size:int
                ):
        self.tokenizer_file = tokenizer_file
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_len = seq_len
        self.batch_size = batch_size
    
    def get_all_sentences(self, ds, lang):
        for item in ds:
            yield item['translation'][lang]
    
    def get_tokenizer(self, ds, lang):
        tokenizer_path = Path(self.tokenizer_file.format(lang))
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not Path.exists(tokenizer_path):
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
            tokenizer.train_from_iterator(self.get_all_sentences(ds, lang), trainer=trainer)
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))

        return tokenizer


    def get_dataset(self, train_pct=0.9):
        lang_config = f"{self.lang_src}-{self.lang_tgt}"
        ds_raw = load_dataset("opus_books", lang_config, split='train')

        # Reduce ds_raw for debugging and testing
        subset_size = len(ds_raw) // 80
        all_indices = list(range(len(ds_raw)))
        subset_indices = random.sample(all_indices, subset_size)
        ds_raw = Subset(ds_raw, subset_indices)

        # Build tokenizer
        tokenizer_src = self.get_tokenizer(ds_raw, self.lang_src)
        tokenizer_tgt = self.get_tokenizer(ds_raw, self.lang_tgt)

        # 90/10 Training/Validation split
        train_ds_size = int(train_pct * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

        # Create BilingualDataset instances
        train_ds = BilingualDataset(train_ds_raw,
                                    tokenizer_src,
                                    tokenizer_tgt,
                                    self.lang_src,
                                    self.lang_tgt,
                                    self.seq_len)
        val_ds = BilingualDataset(val_ds_raw,
                                  tokenizer_src,
                                  tokenizer_tgt,
                                  self.lang_src,
                                  self.lang_tgt,
                                  self.seq_len
                                  )

        # Calculate maximum lengths
        max_len_src = 0
        max_len_tgt = 0

        for item in ds_raw:
            src_ids = tokenizer_src.encode(item['translation'][self.lang_src]).ids
            tgt_ids = tokenizer_tgt.encode(item['translation'][self.lang_tgt]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f"Max length of source sentence: {max_len_src}")
        print(f"Max length of target sentence: {max_len_tgt}")

        # Create DataLoaders
        train_dataloader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

        return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
