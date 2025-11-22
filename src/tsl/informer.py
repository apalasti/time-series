import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.transformer import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .layers.attention import ProbAttention, AttentionLayer
from .layers.embed import DataEmbedding


class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, config: dict):
        super(Informer, self).__init__()
        self.task_name = config["task_type"]
        self.pred_len = config.get("pred_len", -1)
        # self.label_len = configs.label_len

        # Embedding pos_embed_type: timeF
        self.enc_embedding = DataEmbedding(
            config["input_channels"], 
            config["d_model"], 
            config["pos_embed_type"], # Does not matter 
            freq="h",                 # Does not matter
            dropout=config["dropout"],
        )
        # NOTE: Not needed for classification 
        # self.dec_embedding = DataEmbedding(config["input_channels"], config["d_model"], config["pos_embed_type"], configs.freq,
        #                                    config["dropout"])

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, 1, attention_dropout=config["dropout"],
                                      output_attention=False),
                        config["d_model"], config["n_heads"]),
                    config["d_model"],
                    d_ff=2 * config["d_model"],
                    dropout=config["dropout"],
                    activation="gelu"
                ) for l in range(config["n_layers"])
            ],
            # NOTE: Not needed for classification 
            # [
            #     ConvLayer(
            #         config["d_model"]
            #     ) for l in range(configs.e_layers - 1)
            # ] if configs.distil and ('forecast' in self.task_name) else None,
            norm_layer=torch.nn.LayerNorm(config["d_model"])
        )
        # NOTE: Not needed for classification 
        # # Decoder
        # self.decoder = Decoder(
        #     [
        #         DecoderLayer(
        #             AttentionLayer(
        #                 ProbAttention(True, 1, attention_dropout=config["dropout"], output_attention=False),
        #                 config["d_model"], config["n_heads"]),
        #             AttentionLayer(
        #                 ProbAttention(False, 1, attention_dropout=config["dropout"], output_attention=False),
        #                 config["d_model"], config["n_heads"]),
        #             config["d_model"],
        #             d_ff=2 * config["d_model"],
        #             dropout=config["dropout"],
        #             activation="gelu",
        #         )
        #         for l in range(config["n_layers"]//2)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(config["d_model"]),
        #     projection=nn.Linear(config["d_model"], configs.c_out, bias=True)
        # )
        if self.task_name == 'imputation':
            # self.projection = nn.Linear(config["d_model"], configs.c_out, bias=True)
            pass
        if self.task_name == 'anomaly_detection':
            # self.projection = nn.Linear(config["d_model"], configs.c_out, bias=True)
            pass
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(config["dropout"])
            self.projection = nn.Linear(config["d_model"] * config["sequence_len"], config["num_classes"])

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out  # [B, L, D]
    
    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out  # [B, L, D]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        # NOTE: No paddings for us
        # output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
