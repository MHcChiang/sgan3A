from sgan.models import *


class Encoder_Transformer(nn.Module, Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
        )
        self.fc_out = nn.Linear(self.embedding_dim, 2)

    def forward(self, obs_traj):
        return self.transformer(obs_traj)


class Decoder_Transformer(nn.Module, Decoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = nn.Transformer(