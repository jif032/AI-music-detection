"""CNN recurrent models for chunk-level AI music detection.

All models consume segmented log-mel spectrograms with shape
``(batch, segments, 1, mel_bins, segment_frames)`` and return two logits
(``human``, ``AI``). The CNN feature extractor and classification head are
shared; the comparison changes only the temporal encoder and pooling method.
"""

import torch
from torch import nn


class _CNNRecurrentBase(nn.Module):
    """Shared CNN and binary classification head for the three variants."""

    def __init__(
        self,
        n_mels: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        segment_frames: int = 43,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.segment_frames = segment_frames
        self.dropout = dropout

        # Kept identical to the original notebook CNN-GRU feature extractor.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4, 4)),
            # Bottleneck: reduce channels before the recurrent layer.
            nn.Conv2d(128, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Determine the flattened CNN feature size for one segment.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, segment_frames)
            cnn_features = self.cnn(dummy)
            self.rnn_input_size = cnn_features.reshape(1, -1).size(1)

    def _make_classifier(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.hidden_size * 2, 64),  # two directions
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(64, 2),
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode_segments(self, x: torch.Tensor) -> torch.Tensor:
        """Map input ``(B,T,1,H,W)`` to CNN features ``(B,T,F)``."""
        batch, steps, channels, mel_bins, frames = x.shape
        x = x.reshape(batch * steps, channels, mel_bins, frames)
        x = self.cnn(x)
        return x.reshape(batch, steps, -1)


class AudioCRNN(_CNNRecurrentBase):
    """Original CNN with a two-layer BiGRU and temporal mean pooling.

    ``AudioCRNN`` is retained as the original class name for compatibility;
    the experiment/model-selection name is ``cnn_bigru``.
    """

    def __init__(
        self,
        n_mels: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        segment_frames: int = 43,
        dropout: float = 0.3,
    ):
        super().__init__(n_mels, hidden_size, num_layers, segment_frames, dropout)
        self.gru = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = self._make_classifier()
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN features: (B,T,F); BiGRU outputs: (B,T,2H).
        sequence, _ = self.gru(self._encode_segments(x))
        # Preserve the baseline's mean over all temporal steps: (B,2H).
        return self.classifier(sequence.mean(dim=1))


class AudioBiLSTM(_CNNRecurrentBase):
    """Shared CNN, two-layer BiLSTM, temporal mean pooling, same head."""

    def __init__(
        self,
        n_mels: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        segment_frames: int = 43,
        dropout: float = 0.3,
    ):
        super().__init__(n_mels, hidden_size, num_layers, segment_frames, dropout)
        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = self._make_classifier()
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BiLSTM outputs: (B,T,2H); mean pooling gives (B,2H).
        sequence, _ = self.lstm(self._encode_segments(x))
        return self.classifier(sequence.mean(dim=1))


class AudioBiLSTMAttention(_CNNRecurrentBase):
    """Shared CNN/BiLSTM with learned scalar attention over time.

    For BiLSTM output ``H`` of shape ``(B,T,2*hidden_size)``, a learned
    projection scores every time step. Softmax over ``T`` gives normalized
    weights ``(B,T,1)``. Their weighted sum of ``H`` is a context vector of
    shape ``(B,2*hidden_size)``, passed through the same classifier head.
    """

    def __init__(
        self,
        n_mels: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        segment_frames: int = 43,
        dropout: float = 0.3,
    ):
        super().__init__(n_mels, hidden_size, num_layers, segment_frames, dropout)
        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention_score = nn.Linear(hidden_size * 2, 1)
        self.classifier = self._make_classifier()
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(self._encode_segments(x))  # (B,T,2H)
        scores = self.attention_score(torch.tanh(sequence))  # (B,T,1)
        weights = torch.softmax(scores, dim=1)  # normalized across time
        context = torch.sum(sequence * weights, dim=1)  # (B,2H)
        return self.classifier(context)


MODEL_REGISTRY = {
    "cnn_bigru": AudioCRNN,
    "cnn_bilstm": AudioBiLSTM,
    "cnn_bilstm_attention": AudioBiLSTMAttention,
}
# Retain the old identifier so existing notebooks/checkpoint workflows resolve.
MODEL_REGISTRY["cnn_gru"] = AudioCRNN


def build_model(model_type: str, **kwargs) -> nn.Module:
    """Create a temporal model by its explicit experiment name."""
    model_type = model_type.lower()
    try:
        model_class = MODEL_REGISTRY[model_type]
    except KeyError as exc:
        choices = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Unknown model_type={model_type!r}. Choose from: {choices}") from exc
    return model_class(**kwargs)
