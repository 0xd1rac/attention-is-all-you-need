import torch
import src.managers as managers
import src.transformer_components as transformer_components

# if __name__ == "__main__":
#     config_file_path = "./config.json"
#     config = managers.ConfigManager.get_config(config_file_path)
#     print(config)

# config_filepath = "./config.json"
# config = ConfigManager.get_config(config_filepath)

TOKENIZER_FILE = "tokenizers/tokenizer_{0}.json"
EXPERIMENT_NAME = "runs/tmodel"
NUM_EPOCHS = 1 
SEQ_LEN = 600
D_MODEL = 512
LANG_SRC = "en"
LANG_TGT = "it"
MODEL_FOLDER = "weights"
MODEL_BASE_NAME = "tmodel_"
MODEL_BASE_PATH = f"{MODEL_FOLDER}/{MODEL_BASE_NAME}1"
LR = 1e-9
BATCH_SIZE = 8


ds_m = managers.DatasetManager(tokenizer_file = TOKENIZER_FILE,
                      lang_src = LANG_SRC,
                      lang_tgt = LANG_TGT,
                      seq_len = SEQ_LEN,
                      batch_size = BATCH_SIZE
                      )
train_dataloader, val_dataloader, src_lang_tokenizer, tgt_lang_tokenizer = ds_m.get_dataset()


model = managers.ModelManager.build_transformer(src_vocab_size=src_lang_tokenizer.get_vocab_size(),
                                             tgt_vocab_size=tgt_lang_tokenizer.get_vocab_size(),
                                             src_seq_len = SEQ_LEN,
                                             tgt_seq_len = SEQ_LEN
                                             )

managers.ModelManager.train(
    model,
    train_dataloader,
    val_dataloader,
    src_lang_tokenizer,
    tgt_lang_tokenizer, 
    lr=LR,
    model_base_path=MODEL_BASE_PATH,
)