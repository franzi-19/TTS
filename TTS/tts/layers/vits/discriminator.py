import torch
from torch import nn
from torch.nn.modules.conv import Conv1d

from TTS.vocoder.models.hifigan_discriminator import MultiPeriodDiscriminator


class DiscriminatorS(torch.nn.Module):
    """HiFiGAN Scale Discriminator. Channel sizes are different from the original HiFiGAN.

    Args:
        use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        Args:
            x (Tensor): input waveform.

        Returns:
            Tensor: discriminator scores.
            List[Tensor]: list of features from the convolutiona layers.
        """
        feat = []
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feat


class VitsDiscriminator(nn.Module):
    """VITS discriminator wrapping one Scale Discriminator and a stack of Period Discriminator.

    ::
        waveform -> ScaleDiscriminator() -> scores_sd, feats_sd --> append() -> scores, feats
               |--> MultiPeriodDiscriminator() -> scores_mpd, feats_mpd ^

    Args:
        use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        self.sd = DiscriminatorS(use_spectral_norm=use_spectral_norm)
        self.mpd = MultiPeriodDiscriminator(use_spectral_norm=use_spectral_norm)

    def forward(self, x):
        """
        Args:
            x (Tensor): input waveform.

        Returns:
            List[Tensor]: discriminator scores.
            List[List[Tensor]]: list of list of features from each layers of each discriminator.
        """
        scores, feats = self.mpd(x)
        score_sd, feats_sd = self.sd(x)
        return scores + [score_sd], feats + [feats_sd]
