from ModelManager import ModelManager
from TransformerBuilder import TransformerBuilder
from DatasetManager import DatasetManager
from ConfigManager import ConfigManager

config_filepath = "./config.json"
config = ConfigManager.get_config(config_filepath)

model = TransformerBuilder.build_transformer(1000,1000,10,10)
ds_m = DatasetManager(config)
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = ds_m.get_dataset()

ModelManager.train(model=model,
                   train_dataloader=train_dataloader,
                   val_dataloader=val_dataloader,
                   src_lang_tokenizer=tokenizer_src,
                   tgt_lang_tokenizer=tokenizer_tgt,
                   lr = config['lr']
                   )