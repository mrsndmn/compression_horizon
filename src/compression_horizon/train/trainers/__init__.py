"""Concrete trainer implementations: one class per compression-training regime."""

from compression_horizon.train.trainers.base import BaseTrainer
from compression_horizon.train.trainers.compression_head import CompressionHeadTrainer
from compression_horizon.train.trainers.full_cramming import FullCrammingTrainer
from compression_horizon.train.trainers.low_dim import LowDimTrainer
from compression_horizon.train.trainers.prefix_tuning import PrefixTuningTrainer
from compression_horizon.train.trainers.progressive_cramming import ProgressiveCrammingTrainer

__all__ = [
    "BaseTrainer",
    "CompressionHeadTrainer",
    "FullCrammingTrainer",
    "LowDimTrainer",
    "PrefixTuningTrainer",
    "ProgressiveCrammingTrainer",
]
