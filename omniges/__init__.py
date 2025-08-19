"""
Omniges: Multimodal Gesture Generation Framework

A framework for converting audio, text, and other modalities to gesture sequences
Built on top of OmniFlow pipeline with gesture streams replacing image streams
"""

__version__ = "0.1.0"
__author__ = "Omniges Team"

# Import key components for easy access
from .models.omniges_a2g import OmnigesA2GModel, create_omniges_a2g_model, A2GLoss
from .dataloaders.beat_a2g_loader import BeatA2GDataset, create_a2g_dataloader

__all__ = [
    "OmnigesA2GModel",
    "create_omniges_a2g_model", 
    "A2GLoss",
    "BeatA2GDataset",
    "create_a2g_dataloader"
]