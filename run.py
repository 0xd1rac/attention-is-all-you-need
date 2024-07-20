from ModelManager import ModelManager
from DatasetManager import DatasetManager
from ConfigManager import ConfigManager
import torch

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


ds_m = DatasetManager(tokenizer_file = TOKENIZER_FILE,
                      lang_src = LANG_SRC,
                      lang_tgt = LANG_TGT,
                      seq_len = SEQ_LEN,
                      batch_size = BATCH_SIZE
                      )
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = ds_m.get_dataset()


model = ModelManager.build_transformer(src_vocab_size=tokenizer_src.get_vocab_size(),
                                             tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
                                             src_seq_len = SEQ_LEN,
                                             tgt_seq_len = SEQ_LEN
                                             )

model, optimizer, total_epoch_trained = ModelManager.load_model("weights/tmodel_1_1", 
                                                               model, 
                                                               torch.optim.Adam(model.parameters()))

print(f"Loaded a model trained for {total_epoch_trained} epochs.")

sen = "Who are you?"
print(f"Source Sentence: \n{sen}\n")
print(f"Predicted Sentence: \n{ModelManager.run_inference(model, sen, tokenizer_src, tokenizer_tgt)}")

# ModelManager.run_validation(
#     model,
#     val_dataloader,
#     tokenizer_src,
#     tokenizer_tgt
# )
# ModelManager.train(model=model,
#                    train_dataloader=train_dataloader,
#                    val_dataloader=val_dataloader,
#                    src_lang_tokenizer=tokenizer_src,
#                    tgt_lang_tokenizer=tokenizer_tgt,
#                    lr=LR,
#                    model_base_path = MODEL_BASE_PATH,
#                    num_epochs=NUM_EPOCHS
#                    )