# =============================================================================
# models.py
# =============================================================================

import torch
import torch.nn as nn
import logging

log = logging.getLogger(__name__)


# ── CNN Backbone ──────────────────────────────────────────────────────────────


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=(2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
        )

    def forward(self, x):
        return self.block(x)


class CNNClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        ch = config.CNN_CHANNELS

        self.conv1 = ConvBlock(1, ch[0], pool_size=(2, 2))
        self.conv2 = ConvBlock(ch[0], ch[1], pool_size=(2, 2))
        self.conv3 = ConvBlock(ch[1], ch[2], pool_size=(2, 2))
        self.conv4 = ConvBlock(ch[2], ch[3], pool_size=(2, 2))

        # (4, 2) divides (8, 26) evenly — works on MPS
        # gives 8x more spatial info than (1,1)
        self.pool = nn.AdaptiveAvgPool2d((4, 2))

        pool_flat = ch[3] * 4 * 2  # e.g. 64 * 8 = 512
        hidden_dim = min(pool_flat * 2, 512)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pool_flat, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.CNN_DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.CNN_DROPOUT),
            nn.Linear(hidden_dim // 2, 2),
        )

        _init_weights(self)
        _log_model_summary(self, "CNNClassifier")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        return self.classifier(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class CNNBackbone(nn.Module):
    def __init__(self, channels):
        super().__init__()
        in_ch = 1
        blocks = []
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels[-1]

    def forward(self, x):
        return self.blocks(x)


# ── Shared weight initialization ──────────────────────────────────────────────


def _init_weights(module: nn.Module) -> None:
    """
    Standard initialization for CNN with BatchNorm.
    BatchNorm after each conv handles signal normalization automatically —
    no manual weight scaling needed or wanted.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            # NO * 0.1 — BatchNorm normalizes the signal at each layer
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


def _log_model_summary(model: nn.Module, name: str) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"\n{'=' * 55}")
    log.info(f"Model: {name}")
    log.info(f"  Total parameters     : {total:,}")
    log.info(f"  Trainable parameters : {trainable:,}")
    log.info(f"{'=' * 55}")


# ── CNN Classifier ────────────────────────────────────────────────────────────


# class CNNClassifier(nn.Module):
#     """
#     CNN backbone + adaptive global average pooling + MLP head.
#     Input  : (B, 1, n_mels, time_frames)
#     Output : (B, 2) raw logits
#     """

#     def __init__(self, config):
#         super().__init__()
#         self.backbone = CNNBackbone(config.CNN_CHANNELS)
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(config.CNN_DROPOUT),
#             nn.Linear(config.CNN_CHANNELS[-1], config.CNN_CHANNELS[-1] // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(config.CNN_DROPOUT),
#             nn.Linear(config.CNN_CHANNELS[-1] // 2, 2),
#         )
#         # ← THIS IS THE LINE THAT WAS MISSING
#         _init_weights(self)
#         _log_model_summary(self, "CNNClassifier")

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         features = self.backbone(x)
#         pooled = self.pool(features)
#         return self.classifier(pooled)

#     def forward_features(self, x: torch.Tensor) -> torch.Tensor:
#         features = self.backbone(x)
#         pooled = self.pool(features)
#         return pooled.view(pooled.size(0), -1)


# ── CNN + RNN Classifier ──────────────────────────────────────────────────────


class CNNRNNClassifier(nn.Module):
    """
    CNN per segment + RNN over sequence + MLP head.
    Input  : (B, n_segments, 1, n_mels, segment_frames)
    Output : (B, 2) raw logits
    rnn_type: "lstm" | "gru"
    """

    def __init__(self, config, rnn_type: str = "lstm"):
        super().__init__()
        rnn_type = rnn_type.lower()
        assert rnn_type in ("lstm", "gru"), (
            f"rnn_type must be 'lstm' or 'gru', got '{rnn_type}'"
        )

        self.rnn_type = rnn_type
        self.backbone = CNNBackbone(config.CNN_CHANNELS)
        self.segment_pool = nn.AdaptiveAvgPool2d((1, 1))
        cnn_out_dim = config.CNN_CHANNELS[-1]

        RNNClass = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = RNNClass(
            input_size=cnn_out_dim,
            hidden_size=config.RNN_HIDDEN_SIZE,
            num_layers=config.RNN_NUM_LAYERS,
            batch_first=True,
            dropout=(config.RNN_DROPOUT if config.RNN_NUM_LAYERS > 1 else 0.0),
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(config.CNN_DROPOUT),
            nn.Linear(config.RNN_HIDDEN_SIZE, config.RNN_HIDDEN_SIZE // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.CNN_DROPOUT),
            nn.Linear(config.RNN_HIDDEN_SIZE // 2, 2),
        )
        # ← THIS IS THE LINE THAT WAS MISSING
        _init_weights(self)
        _log_model_summary(self, f"CNN+{rnn_type.upper()}Classifier")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        features = self.backbone(x_flat)
        pooled = self.segment_pool(features)
        pooled = pooled.view(B, T, -1)

        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(pooled)
        else:
            _, h_n = self.rnn(pooled)

        last_hidden = h_n[-1]
        return self.classifier(last_hidden)


# ── Factory ───────────────────────────────────────────────────────────────────


def build_model(config, model_type: str | None = None) -> nn.Module:
    mtype = (model_type or config.MODEL_TYPE).lower()
    log.info(f"Building model: {mtype}")
    if mtype == "cnn":
        return CNNClassifier(config)
    elif mtype == "cnn_lstm":
        return CNNRNNClassifier(config, rnn_type="lstm")
    elif mtype == "cnn_gru":
        return CNNRNNClassifier(config, rnn_type="gru")
    else:
        raise ValueError(
            f"Unknown model_type='{mtype}'. Choose from: cnn, cnn_lstm, cnn_gru"
        )


def get_model_input_shape(config, model_type: str | None = None) -> tuple:
    mtype = (model_type or config.MODEL_TYPE).lower()
    if mtype == "cnn":
        return (1, config.N_MELS, config.SPEC_TIME_FRAMES)
    elif mtype in ("cnn_lstm", "cnn_gru"):
        segment_frames = 43
        n_segments = config.SPEC_TIME_FRAMES // segment_frames
        return (n_segments, 1, config.N_MELS, segment_frames)
    elif mtype == "ast":
        return (1, config.CHUNK_SEC * 16000)
    else:
        raise ValueError(f"Unknown model_type='{mtype}'")
