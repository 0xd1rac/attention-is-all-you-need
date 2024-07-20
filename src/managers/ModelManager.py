from src.imports.common_imports import *
import src.transformer_components as transformer_components
import src.data as data

class ModelManager():
    @staticmethod
    def build_transformer(src_vocab_size: int,
                          tgt_vocab_size: int,
                          src_seq_len: int,
                          tgt_seq_len: int,
                          d_model: int = 512,
                          N: int = 6,  # Number of encoder and decoder blocks
                          h: int = 8,  # Number of heads in MHA block
                          dropout_proba: float = 0.1,
                          d_ff: int = 2048  # Number of parameters in the Feed Forward Layer
                          ) -> transformer_components.Transformer:
        """
        Build a Transformer model.

        Args:
            src_vocab_size (int): Size of the source vocabulary.
            tgt_vocab_size (int): Size of the target vocabulary.
            src_seq_len (int): Length of the source sequence.
            tgt_seq_len (int): Length of the target sequence.
            d_model (int): Dimension of the model. Default is 512.
            N (int): Number of encoder and decoder blocks. Default is 6.
            h (int): Number of heads in the multi-head attention block. Default is 8.
            dropout_proba (float): Dropout probability for regularization. Default is 0.1.
            d_ff (int): Number of parameters in the feed-forward layer. Default is 2048.

        Returns:
            Transformer: The constructed Transformer model.
        """
        # Create the embedding layers
        src_embed =  transformer_components.InputEmbeddings(d_model, src_vocab_size)
        tgt_embed =  transformer_components.InputEmbeddings(d_model, tgt_vocab_size)

        # Create the positional encoding layers
        src_pos =  transformer_components.PositionalEncoding(d_model, src_seq_len, dropout_proba)
        tgt_pos =  transformer_components.PositionalEncoding(d_model, tgt_seq_len, dropout_proba)

        # Create the encoder blocks
        encoder_blocks = []
        for _ in range(N):
            self_attention_block = transformer_components.MultiHeadAttentionBlock(d_model, h, dropout_proba)
            feed_forward_block = transformer_components.FeedForwardBlock(d_model, d_ff, dropout_proba)
            encoder_block = transformer_components.EncoderBlock(self_attention_block, feed_forward_block, dropout_proba)
            encoder_blocks.append(encoder_block)

        # Create the decoder blocks
        decoder_blocks = []
        for _ in range(N):
            self_attention_block = transformer_components.MultiHeadAttentionBlock(d_model, h, dropout_proba)
            cross_attention_block = transformer_components.MultiHeadAttentionBlock(d_model, h, dropout_proba)
            feed_forward_block = transformer_components.FeedForwardBlock(d_model, d_ff, dropout_proba)
            decoder_block = transformer_components.DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout_proba)
            decoder_blocks.append(decoder_block)

        # Create encoder and decoder
        encoder =  transformer_components.Encoder(nn.ModuleList(encoder_blocks))
        decoder =  transformer_components.Decoder(nn.ModuleList(decoder_blocks))

        # Create the projection layer
        projection_layer =  transformer_components.ProjectionLayer(d_model, tgt_vocab_size)

        # Create the transformer
        transformer =  transformer_components.Transformer(encoder=encoder,
                                  decoder=decoder,
                                  src_embed=src_embed,
                                  tgt_embed=tgt_embed,
                                  src_pos=src_pos,
                                  tgt_pos=tgt_pos,
                                  projection_layer=projection_layer)

        # Initialize the parameters
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer

    @staticmethod
    def load_model(model_file_path: str, 
                   model: nn.Module, 
                   optimizer: torch.optim.Optimizer = None
                   ):
        checkpoint = torch.load(model_file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_epoch_trained = checkpoint.get('total_epoch_trained', 0)
        return model, optimizer, total_epoch_trained

    @staticmethod
    def save_model(model: nn.Module, 
                   epoch_trained: int,
                   optimizer: torch.optim.Optimizer,
                   model_file_path: str
                  ):
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_epoch_trained': epoch_trained
        }
        print(f"[INFO] Saving {model_file_path}")
        torch.save(save_dict, model_file_path)
        pass

    @staticmethod
    def train(model: nn.Module,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              src_lang_tokenizer: Tokenizer,
              tgt_lang_tokenizer: Tokenizer,
              lr: float,
              model_base_path: str,
              epoch_trained: int = 0, # should be 0 if the model is not preloaded
              num_epochs: int = 5,
              device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ) -> None:
        print(f"[INFO] Training using device: {device}")
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(),lr=lr,eps=1e-9)
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
                proj_output = model.project(decoder_output) # (batch_size, seq_len, tgt_vocab_size)
                
                label = batch['label'].to(device) # (batch_size, seq_len)
                
                """
                proj_output -> (batch_size, seq_len, tgt_vocab_size)
                after, project_output -> (batch_size * seq_len, tgt_vocab_size)

                label -> (batch_size, seq_len)
                after , label -> (batch_size * seq_len)
                
                """

                loss = loss_fn(proj_output.view(-1, tgt_lang_tokenizer.get_vocab_size()), 
                               label.view(-1)
                               )

                batch_iterator.set_postfix({
                    "loss" : f"{loss.item():6.3f}"
                })


                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
            
            epoch_trained += 1 
            # save the model at the end of every epoch
            ModelManager.save_model(model,epoch_trained, optimizer,f"{model_base_path}_{epoch_trained}")
        print(f"Training has been completed for {num_epochs}")

    @staticmethod 
    def greedy_decode(model, 
                      encoder_input, 
                      encoder_mask,
                      src_lang_tokenizer,
                      tgt_lang_tokenizer, 
                      max_len,
                      device
                      ):
        sos_symbol_idx = tgt_lang_tokenizer.token_to_id('[SOS]')
        eos_symbol_idx = tgt_lang_tokenizer.token_to_id('[EOS]')

        encoder_output = model.encode(encoder_input, encoder_mask)

        # initialize the decoder input with the sos token
        decoder_input = torch.empty(1,1).fill_(sos_symbol_idx).type_as(encoder_input).to(device)
        
        while True:
            # if the we reach max len of symbols to generate, break 
            if decoder_input.size(1) == max_len:
                break
            
            # build mask for decoder - casual mask 
            decoder_mask = data.causal_mask(decoder_input.size(1)).type_as(encoder_input).to(device)

            # calculate output
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)

            # get next token
            prob = model.project(decoder_output[:, -1])
            _, next_token = torch.max(prob, dim=1)

            # append the new token to the input
            decoder_input = torch.cat(
                [
                    decoder_input, 
                    torch.empty(1,1)
                        .type_as(encoder_input)
                        .fill_(next_token.item())
                        .to(device)
                ],
                dim=1
            )

            if next_token == eos_symbol_idx:
                break
        
        return decoder_input.squeeze(0)


    @staticmethod
    def run_validation(model: nn.Module,
                      val_dataloader: DataLoader,
                      src_lang_tokenizer: Tokenizer,
                      tgt_lang_tokenizer: Tokenizer,
                      max_len: int = 250, 
                      device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                      ) -> None:
        print(f"[INFO] Running inference on validation dataset.")

        src_text_lis, tgt_text_lis, model_output_text_lis = [], [], []
        
        with torch.no_grad():
            model.eval()
            for batch in val_dataloader:
                encoder_input = batch['encoder_input'].to(device)
                encoder_mask =  batch['encoder_mask'].to(device)
            
                assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

                model_output = ModelManager.greedy_decode(model, encoder_input, encoder_mask, src_lang_tokenizer, tgt_lang_tokenizer, max_len, device)

                src_text = batch['src_text'][0]
                tgt_text = batch['tgt_text'][0]
                model_output_text = tgt_lang_tokenizer.decode(model_output.detach().cpu().numpy())
                
                src_text_lis.append(src_text)
                tgt_text_lis.append(tgt_text)
                model_output_text_lis.append(model_output_text)

                # print
                # print("-" * 20)
                # print(f"Source: {src_text}\n")
                # print(f"Target: {tgt_text}\n")
                # print(f"Model Predicted: {model_output_text}\n")
                # print("-" * 20)
        

        metric_dict = {
            "cer": MetricManager.get_metric("cer", model_output_text_lis, tgt_text_lis),
            "wer": MetricManager.get_metric("wer", model_output_text_lis, tgt_text_lis),
            "bleu": MetricManager.get_metric("bleu", model_output_text_lis, tgt_text_lis)
        }

        print("--------- METRICS ------------")
        print(f"Char Error Rate: {metric_dict['cer']}")
        print(f"Word Error Rate: {metric_dict['wer']}")
        print(f"BLEU: {metric_dict['bleu']}")
        print("------------------------------")

        return metric_dict


    @staticmethod
    def run_inference(model: nn.Module,
                      sentence_in_src_lang: str,
                      src_lang_tokenizer: Tokenizer,
                      tgt_lang_tokenizer: Tokenizer,
                      max_len: int = 20,
                      device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                     ) -> str:
        model.eval()
        with torch.no_grad():
            # Transform the sentence into tokens 
            encoder_input_token_ids = src_lang_tokenizer.encode(sentence_in_src_lang).ids
            sos_token_id = src_lang_tokenizer.token_to_id("[SOS]")
            eos_token_id = src_lang_tokenizer.token_to_id("[EOS]")
            padding_token_id = src_lang_tokenizer.token_to_id("[PAD]")
            encoder_input = torch.tensor(
                                [sos_token_id] + encoder_input_token_ids + [eos_token_id],
                                dtype=torch.int64
                                ).unsqueeze(0).to(device)

            encoder_mask = (encoder_input != src_lang_tokenizer.token_to_id("[PAD]")).unsqueeze(1).to(device)

            decoded_output = ModelManager.greedy_decode(model, 
                                                        encoder_input, 
                                                        encoder_mask, 
                                                        src_lang_tokenizer, 
                                                        tgt_lang_tokenizer, 
                                                        max_len, 
                                                        device
                                                        )
            
            output_sentence_in_tgt_lang = tgt_lang_tokenizer.decode(decoded_output.detach().cpu().numpy())

            return output_sentence_in_tgt_lang
