from typing import Optional
import torch
from torch import Tensor

from ..core.transforms_interface import BaseWaveformTransform
from ..utils.dsp import calculate_rms
from ..utils.io import Audio
from ..utils.object_dict import ObjectDict


class Mix(BaseWaveformTransform):
    """
    Create a new sample by mixing it with another random sample from the same batch

    Signal-to-noise ratio (where "noise" is the second random sample) is selected
    randomly between `min_snr_in_db` and `max_snr_in_db`.

    `mix_target` controls how resulting targets are generated. It can be one of
    "original" (targets are those of the original sample) or "union" (targets are the
    union of original and overlapping targets)

    """

    supported_modes = {"per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = False

    supports_target = True
    requires_target = False

    def __init__(
        self,
        min_snr_in_db: float = 0.0,
        max_snr_in_db: float = 5.0,
        min_len_ratio: float = 0.1,
        max_len_ratio: float = 0.5,
        mix_target: str = "union",
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")

        self.min_len_ratio = min_len_ratio
        self.max_len_ratio = max_len_ratio
        if self.min_len_ratio > self.max_len_ratio and self.min_len_ratio < 1.0:
            raise ValueError("min_len_ratio must not be greater than max_len_ratio")

        self.mix_target = mix_target
        if mix_target == "original":
            self._mix_target = lambda target, background_target, snr: target

        elif mix_target == "union":
            self._mix_target = lambda target, background_target, snr: torch.maximum(
                target, background_target
            )

        else:
            raise ValueError("mix_target must be one of 'original' or 'union'.")

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):

        batch_size, num_channels, num_samples = samples.shape
        snr_distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.min_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            ),
            high=torch.tensor(
                self.max_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            ),
            validate_args=True,
        )

        # randomize SNRs
        self.transform_parameters["snr_in_db"] = snr_distribution.sample(
            sample_shape=(batch_size,)
        )

        # randomize index of second sample
        self.transform_parameters["sample_idx"] = torch.randint(
            0,
            batch_size,
            (batch_size,),
            device=samples.device,
        )

        # randomize noise segments
        if self.min_len_ratio < 1.0:
            self.transform_parameters["segment_len"] = torch.randint(
                int(num_samples * self.min_len_ratio),
                int(num_samples * self.max_len_ratio),
                (batch_size,),
            ).tolist()

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        snr = self.transform_parameters["snr_in_db"]
        idx = self.transform_parameters["sample_idx"]

        background_samples = Audio.rms_normalize(samples[idx])
        background_rms = calculate_rms(samples) / (10 ** (snr.unsqueeze(dim=-1) / 20))
        background_samples = background_rms.unsqueeze(-1) * background_samples

        if self.min_len_ratio < 1.0:
            mixed_samples = samples.clone()
            for i, l in enumerate(self.transform_parameters["segment_len"]):
                s1 = torch.randint(0, samples.shape[2] - l, (1,))
                s2 = torch.randint(0, samples.shape[2] - l, (1,))
                mixed_samples[i, :, s1 : s1 + l] += background_samples[i, :, s2 : s2 + l]
        else:
            mixed_samples = samples + background_samples

        if targets is None:
            mixed_targets = None

        else:
            background_targets = targets[idx]
            mixed_targets = self._mix_target(targets, background_targets, snr)

        return ObjectDict(
            samples=mixed_samples,
            sample_rate=sample_rate,
            targets=mixed_targets,
            target_rate=target_rate,
        )
