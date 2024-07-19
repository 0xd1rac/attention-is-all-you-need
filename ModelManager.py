from ManagerImports import * 
from ConfigManager import ConfigManager
from DatasetManager import DatasetManager
from TransformerBuilder import TransformerBuilder
# sys.path.append(os.path.abspath(os.path.join('..', 'Transformer')))


class ModelManager():
    @staticmethod
    def load_model(model_file_path: str): 
        pass

    @staticmethod
    def save_model(model_file_path: str):
        pass

    @staticmethod
    def train(
             model,
             train_dataloader,
             val_dataloader,
             src_lang_tokenizer,
             tgt_lang_tokenizer,
             lr,
             device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
             num_epochs = 5,
            
             ):


        print(f"[INFO] Using device: {device}")
        model.to(device)

        initial_epoch = 0
        global_step = 0 

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     eps=1e-9
                                     )

        pad_symbol = '[PAD]'
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=src_lang_tokenizer.token_to_id(pad_symbol),
            label_smoothing=0.1
        ).to(device)


        for epoch in range(0, num_epochs):
            model.train()
            batch_iterator = tqdm(train_dataloader,
                                  desc=f"Processing epoch {epoch:02d}"
                                  )
            for batch in batch_iterator:
                encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
                encoder_mask = batch['encoder_mask'].to(device)
                
                decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
                decoder_mask = batch['decoder_mask'].to(device)

                encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #(batch_size, seq_len, d_model)
                proj_output = model.project(decoder_output)

                label = batch['label'].to(device) # batch_size, seq_len)
                
                # (batch_size, seq_len, tgt_vocab_size) -> (batch_size, seq_len, tgt_vocab_size)
                loss = loss_fn(proj_output.view(-1, tokenizer_src.get_vocab_size()), label.view(-1))

                batch_iterator.set_postfix({
                    "loss" : f"{loss.item():6.3f}"
                })


                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1 

        print(f"Training has been completed for {num_epochs}")
        


