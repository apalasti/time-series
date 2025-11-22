import torch
import torch.nn as nn
from .layers.transformer import Decoder, DecoderLayer, Encoder, EncoderLayer
from .layers.attention import DSAttention, AttentionLayer
from .layers.embed import DataEmbedding
import torch.nn.functional as F


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class NSTransformer(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(self, config: dict):
        super(NSTransformer, self).__init__()
        self.task_name = config["task_type"]
        self.pred_len = config.get("pred_len", -1)
        self.seq_len = config["sequence_len"]
        self.label_len = config.get("label_len", -1)

        self.p_hidden_layers = 2
        self.p_hidden_dims = [256, 256]

        # Embedding
        self.enc_embedding = DataEmbedding(
            config["input_channels"], 
            config["d_model"], 
            config["pos_embed_type"], # Does not matter
            "h",                      # Does not matter
            config["dropout"],
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, 1, attention_dropout=config["dropout"],
                                    output_attention=False), config["d_model"], config["n_heads"]),
                    config["d_model"],
                    d_ff=2 * config["d_model"],
                    dropout=config["dropout"],
                    activation="gelu",
                ) for l in range(config["n_layers"])
            ],
            norm_layer=torch.nn.LayerNorm(config["d_model"])
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            pass
            # self.dec_embedding = DataEmbedding(configs.dec_in, config["d_model"], configs.embed, configs.freq,
            #                                    config["dropout"])
            # self.decoder = Decoder(
            #     [
            #         DecoderLayer(
            #             AttentionLayer(
            #                 DSAttention(True, 1, attention_dropout=config["dropout"],
            #                             output_attention=False),
            #                 config["d_model"], config["n_heads"]),
            #             AttentionLayer(
            #                 DSAttention(False, 1, attention_dropout=config["dropout"],
            #                             output_attention=False),
            #                 config["d_model"], config["n_heads"]),
            #             config["d_model"],
            #             d_ff=2 * config["d_model"],
            #             dropout=config["dropout"],
            #             activation="gelu",
            #         )
            #         for l in range(config["n_layers"])
            #     ],
            #     norm_layer=torch.nn.LayerNorm(config["d_model"]),
            #     projection=nn.Linear(config["d_model"], configs.c_out, bias=True)
            # )
        if self.task_name == 'imputation':
            pass
            # self.projection = nn.Linear(config["d_model"], configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            pass
            # self.projection = nn.Linear(config["d_model"], configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(config["dropout"])
            self.projection = nn.Linear(config["d_model"] * config["sequence_len"], config["num_classes"])

        self.tau_learner = Projector(
            enc_in=config["input_channels"], seq_len=config["sequence_len"], hidden_dims=self.p_hidden_dims,
            hidden_layers=self.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(
            enc_in=config["input_channels"], seq_len=config["sequence_len"],
            hidden_dims=self.p_hidden_dims, hidden_layers=self.p_hidden_layers,
            output_dim=config["sequence_len"])

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc)
        threshold = 80.0
        tau_clamped = torch.clamp(tau, max=threshold)  # avoid numerical overflow
        tau = tau_clamped.exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)

        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, tau=tau, delta=delta)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        mean_enc = mean_enc.unsqueeze(1).detach()
        x_enc = x_enc - mean_enc
        x_enc = x_enc.masked_fill(mask == 0, 0)
        std_enc = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        std_enc = std_enc.unsqueeze(1).detach()
        x_enc /= std_enc
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc)
        threshold = 80.0
        tau_clamped = torch.clamp(tau, max=threshold)  # avoid numerical overflow
        tau = tau_clamped.exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.projection(enc_out)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def anomaly_detection(self, x_enc):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc)
        threshold = 80.0
        tau_clamped = torch.clamp(tau, max=threshold)  # avoid numerical overflow
        tau = tau_clamped.exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)
        # embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.projection(enc_out)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        std_enc = torch.sqrt(
            torch.var(x_enc - mean_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc)
        threshold = 80.0
        tau_clamped = torch.clamp(tau, max=threshold)  # avoid numerical overflow
        tau = tau_clamped.exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)
        # embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        # NOTE: We have no paddings
        # output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, L, D]
        return None
