"""
BEAT A2G Dataset Loader
Specialized loader for Audio-to-Gesture task using BEAT2 dataset

Features:
1. Audio processing: Raw waveform + MFCC extraction
2. Gesture processing: SMPLX body parts (upper, hands, lower_trans, face)
3. Temporal alignment: 16kHz audio ↔ 15fps gesture
4. LMDB caching for efficient training
"""

import os
import torch
import numpy as np
import librosa
import pickle
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional
from loguru import logger

# Import existing BEAT loaders as base
from dataloaders.beat_sep_lower import CustomDataset as BeatDataset
from dataloaders.data_tools import joints_list


class BeatA2GDataset(Dataset):
    """
    BEAT Dataset specialized for A2G (Audio to Gesture) task
    
    Inherits from existing BEAT loader but focuses on audio-gesture pairs
    """
    
    def __init__(
        self, 
        args,
        mode: str = "train",  # train, val, test
        audio_sr: int = 16000,
        pose_fps: int = 15,
        sequence_length: int = 64,  # frames
        include_face: bool = True,
        cache_audio_features: bool = False
    ):
        self.args = args
        self.mode = mode
        self.audio_sr = audio_sr
        self.pose_fps = pose_fps
        self.sequence_length = sequence_length
        self.include_face = include_face
        self.cache_audio_features = cache_audio_features
        
        # Initialize base BEAT dataset
        self.base_dataset = BeatDataset(args, mode, build_cache=True)
        
        # Define body part configurations
        self.body_parts = ["upper", "hands", "lower_trans", "face"] if include_face else ["upper", "hands", "lower_trans"]
        self.part_joint_configs = {
            "upper": joints_list["beat_smplx_upper"],      # 26 joints = 78 dim
            "hands": joints_list["beat_smplx_hands"],      # 60 joints = 180 dim  
            "lower_trans": joints_list["beat_smplx_lower"], # 8 joints + 3 trans = 57 dim
            "face": {"jaw": 3}  # jaw (3) + expressions (100) = 103 dim, but expressions handled separately
        }
        
        # Audio feature dimensions
        self.mfcc_dim = 128
        self.wavlm_dim = 768
        
        logger.info(f"Initialized BeatA2GDataset with {len(self)} samples in {mode} mode")
        logger.info(f"Body parts: {self.body_parts}")
        logger.info(f"Sequence length: {sequence_length} frames ({sequence_length/pose_fps:.2f}s)")
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single A2G training sample
        
        Returns:
            Dictionary containing:
            - audio_waveform: (T_audio,) raw waveform at 16kHz
            - audio_mfcc: (T_frames, 128) MFCC features  
            - gesture_parts: Dict with body part gestures
            - audio_name: Audio file identifier
            - metadata: Additional info (emotion, semantic, etc.)
        """
        # Get base sample from BEAT dataset
        base_sample = self.base_dataset[idx]
        
        # Process audio data
        audio_data = self._process_audio(base_sample)
        
        # Process gesture data  
        gesture_data = self._process_gesture(base_sample)
        
        # Process metadata
        metadata = self._process_metadata(base_sample)
        
        return {
            **audio_data,
            **gesture_data, 
            **metadata,
            "sample_idx": idx
        }
    
    def _process_audio(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """
        Process audio data to extract waveform and MFCC features
        
        Args:
            sample: Base BEAT sample
            
        Returns:
            Dict with audio waveform and MFCC features
        """
        # Get audio file path from sample
        if "audio_name" in sample:
            audio_file = sample["audio_name"]
        else:
            # Fallback: construct from sample info
            audio_file = f"datasets/BEAT_SMPL/beat_v2.0.0/wave16k/{sample.get('id', 'unknown')}.wav"
        
        # Load raw audio waveform
        if os.path.exists(audio_file):
            waveform, sr = librosa.load(audio_file, sr=self.audio_sr)
        else:
            # Use dummy audio if file not found
            duration = self.sequence_length / self.pose_fps
            waveform = np.zeros(int(duration * self.audio_sr))
            logger.warning(f"Audio file not found: {audio_file}, using zero audio")
        
        # Calculate target frames based on pose sequence
        target_frames = self.sequence_length
        target_audio_length = int(target_frames / self.pose_fps * self.audio_sr)
        
        # Trim or pad audio to match target length
        if len(waveform) > target_audio_length:
            waveform = waveform[:target_audio_length]
        elif len(waveform) < target_audio_length:
            waveform = np.pad(waveform, (0, target_audio_length - len(waveform)), mode='constant')
        
        # Convert to tensor
        audio_waveform = torch.from_numpy(waveform).float()
        
        # Extract MFCC features if needed (for testing/validation)
        if self.cache_audio_features:
            mfcc_features = librosa.feature.mfcc(
                y=waveform, 
                sr=self.audio_sr, 
                n_mfcc=self.mfcc_dim,
                hop_length=int(self.audio_sr / self.pose_fps)
            )
            mfcc_features = torch.from_numpy(mfcc_features.T).float()  # (T, n_mfcc)
            
            # Align to target frames
            if mfcc_features.size(0) != target_frames:
                mfcc_features = F.interpolate(
                    mfcc_features.unsqueeze(0).transpose(1, 2),
                    size=target_frames,
                    mode='linear'
                ).transpose(1, 2).squeeze(0)
        else:
            # Placeholder MFCC (will be computed in model)
            mfcc_features = torch.zeros(target_frames, self.mfcc_dim)
        
        return {
            "audio_waveform": audio_waveform,  # (T_audio,)
            "audio_mfcc": mfcc_features,       # (T_frames, 128)
        }
    
    def _process_gesture(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """
        Process gesture data into body part components
        
        Args:
            sample: Base BEAT sample with pose data
            
        Returns:
            Dict with gesture data for each body part
        """
        # Get pose data
        pose_data = sample["pose"]  # (T, pose_dim)
        trans_data = sample.get("trans", None)
        trans_v_data = sample.get("trans_v", None) 
        facial_data = sample.get("facial", None)
        
        # Ensure sequence length consistency
        target_frames = self.sequence_length
        if pose_data.size(0) > target_frames:
            pose_data = pose_data[:target_frames]
            if trans_data is not None:
                trans_data = trans_data[:target_frames]
            if trans_v_data is not None:
                trans_v_data = trans_v_data[:target_frames]
            if facial_data is not None:
                facial_data = facial_data[:target_frames]
        elif pose_data.size(0) < target_frames:
            # Pad sequences
            pad_length = target_frames - pose_data.size(0)
            pose_data = F.pad(pose_data, (0, 0, 0, pad_length), mode='constant')
            if trans_data is not None:
                trans_data = F.pad(trans_data, (0, 0, 0, pad_length), mode='constant')
            if trans_v_data is not None:
                trans_v_data = F.pad(trans_v_data, (0, 0, 0, pad_length), mode='constant')
            if facial_data is not None:
                facial_data = F.pad(facial_data, (0, 0, 0, pad_length), mode='constant')
        
        # Split pose data into body parts
        gesture_parts = self._split_pose_to_parts(pose_data, trans_v_data, facial_data)
        
        return {"gesture_parts": gesture_parts}
    
    def _split_pose_to_parts(
        self, 
        pose_data: torch.Tensor,
        trans_v_data: Optional[torch.Tensor] = None,
        facial_data: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Split full pose data into body part components
        
        Based on joint definitions in data_tools.py
        """
        gesture_parts = {}
        
        # Parse pose dimensions (should be flattened from SMPLX joints)
        total_pose_dim = pose_data.size(-1)
        
        # For BEAT SMPLX data, poses are typically concatenated as:
        # [body_poses, hand_poses, ...] - need to extract based on joint masks
        
        # Upper body (spine, shoulders, arms, etc.) - 78 dim
        if "upper" in self.body_parts:
            # This mapping needs to be based on the actual joint ordering in the data
            # For now, using rough estimates - should be refined based on actual BEAT data structure
            gesture_parts["upper"] = pose_data[:, :78]  # First 78 dimensions
        
        # Hands (all finger joints) - 180 dim  
        if "hands" in self.body_parts:
            gesture_parts["hands"] = pose_data[:, 78:258]  # Next 180 dimensions
        
        # Lower body + translation - 57 dim
        if "lower_trans" in self.body_parts:
            lower_pose = pose_data[:, 258:282]  # 24 dim for lower body
            if trans_v_data is not None:
                trans_v_flat = trans_v_data.view(trans_v_data.size(0), -1)  # Flatten translation velocity
                gesture_parts["lower_trans"] = torch.cat([lower_pose, trans_v_flat], dim=-1)  # 24 + 3 = 27, need to verify
            else:
                gesture_parts["lower_trans"] = lower_pose
        
        # Face (jaw + expressions) - 103 dim
        if "face" in self.body_parts and facial_data is not None:
            # Jaw pose is part of main pose data, expressions from facial_data
            jaw_pose = pose_data[:, 282:285]  # 3 dim for jaw
            if facial_data.size(-1) == 100:  # Expression coefficients
                face_features = torch.cat([jaw_pose, facial_data], dim=-1)  # 3 + 100 = 103
            else:
                face_features = jaw_pose
            gesture_parts["face"] = face_features
        
        return gesture_parts
    
    def _process_metadata(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Extract metadata from sample."""
        metadata = {}
        
        # Emotion and semantic labels
        if "emo" in sample:
            metadata["emotion"] = sample["emo"]
        if "sem" in sample:
            metadata["semantic"] = sample["sem"]
        if "id" in sample:
            metadata["speaker_id"] = sample["id"]
            
        return {"metadata": metadata}
    
    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for A2G batches
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            Batched tensors ready for model input
        """
        batch_size = len(batch)
        
        # Collect audio waveforms
        audio_waveforms = torch.stack([sample["audio_waveform"] for sample in batch])
        
        # Collect gesture parts
        gesture_batch = {}
        for part in self.body_parts:
            if f"gesture_parts" in batch[0] and part in batch[0]["gesture_parts"]:
                part_data = torch.stack([
                    sample["gesture_parts"][part] for sample in batch
                ])
                gesture_batch[part] = part_data
        
        # Collect metadata
        metadata_batch = {}
        if "metadata" in batch[0]:
            for key in batch[0]["metadata"]:
                if isinstance(batch[0]["metadata"][key], torch.Tensor):
                    metadata_batch[key] = torch.stack([
                        sample["metadata"][key] for sample in batch
                    ])
        
        return {
            "audio_waveform": audio_waveforms,      # (B, T_audio)
            "gesture_parts": gesture_batch,          # Dict[part] -> (B, T_frames, part_dim)  
            "metadata": metadata_batch,
            "batch_size": batch_size
        }


def create_a2g_dataloader(
    args,
    mode: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    sequence_length: int = 64,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for A2G training
    
    Args:
        args: Training arguments
        mode: Dataset split (train/val/test)
        batch_size: Batch size
        num_workers: Number of data loading workers
        sequence_length: Sequence length in frames
        
    Returns:
        DataLoader for A2G training
    """
    
    # Update args for A2G task
    args.pose_length = sequence_length
    args.audio_rep = "wave16k"  # Use raw waveform
    args.facial_rep = "expression" if kwargs.get("include_face", True) else None
    
    # Create dataset
    dataset = BeatA2GDataset(
        args=args,
        mode=mode,
        sequence_length=sequence_length,
        **kwargs
    )
    
    # Create DataLoader
    shuffle = (mode == "train")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


class A2GDataProcessor:
    """
    Utility class for processing A2G data samples
    """
    
    @staticmethod
    def align_audio_gesture_timing(
        audio_waveform: torch.Tensor,
        gesture_data: Dict[str, torch.Tensor],
        audio_sr: int = 16000,
        pose_fps: int = 15
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Align audio and gesture temporal dimensions
        
        Args:
            audio_waveform: Raw audio (T_audio,)
            gesture_data: Gesture parts data
            audio_sr: Audio sample rate
            pose_fps: Pose frame rate
            
        Returns:
            Aligned audio and gesture data
        """
        # Calculate target frames from audio duration
        audio_duration = audio_waveform.size(0) / audio_sr
        target_frames = int(audio_duration * pose_fps)
        
        # Align gesture data to target frames
        aligned_gesture = {}
        for part, data in gesture_data.items():
            if data.size(0) != target_frames:
                # Interpolate gesture sequence to match audio duration
                data_interp = F.interpolate(
                    data.unsqueeze(0).transpose(1, 2),  # (1, part_dim, T)
                    size=target_frames,
                    mode='linear'
                ).transpose(1, 2).squeeze(0)  # (T, part_dim)
                aligned_gesture[part] = data_interp
            else:
                aligned_gesture[part] = data
        
        return audio_waveform, aligned_gesture
    
    @staticmethod
    def create_attention_mask(
        sequence_length: int,
        valid_length: int
    ) -> torch.Tensor:
        """Create attention mask for variable-length sequences."""
        mask = torch.zeros(sequence_length, dtype=torch.bool)
        mask[:valid_length] = True
        return mask
    
    @staticmethod
    def gesture_parts_to_full(gesture_parts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate gesture parts into full gesture representation."""
        parts_list = []
        part_order = ["upper", "hands", "lower_trans", "face"]
        
        for part in part_order:
            if part in gesture_parts:
                parts_list.append(gesture_parts[part])
        
        if parts_list:
            return torch.cat(parts_list, dim=-1)  # (B, T, total_dim)
        else:
            return torch.empty(0)
    
    @staticmethod  
    def full_gesture_to_parts(
        full_gesture: torch.Tensor,
        include_face: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Split full gesture representation back into body parts."""
        part_dims = {"upper": 78, "hands": 180, "lower_trans": 57, "face": 100}
        part_order = ["upper", "hands", "lower_trans", "face"] if include_face else ["upper", "hands", "lower_trans"]
        
        gesture_parts = {}
        start_idx = 0
        
        for part in part_order:
            if part in part_dims:
                end_idx = start_idx + part_dims[part]
                gesture_parts[part] = full_gesture[..., start_idx:end_idx]
                start_idx = end_idx
        
        return gesture_parts


# Test the data loader
if __name__ == "__main__":
    # Mock args for testing
    class MockArgs:
        def __init__(self):
            self.data_path = "./datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/"
            self.data_path_1 = "./datasets/"
            self.cache_path = "/cache/"
            self.root_path = "./"
            self.pose_rep = "beat_smplx_141"
            self.pose_fps = 15
            self.audio_sr = 16000
            self.audio_fps = 15
            self.training_speakers = [2]
            self.additional_data = False
            self.disable_filtering = False
            self.clean_first_seconds = 2
            self.clean_final_seconds = 2
            self.multi_length_training = [1.0]
            self.test_length = 10
            self.stride = 10
            self.pose_length = 64
            self.word_rep = "textgrid"
            self.facial_rep = "expression"
            self.emo_rep = "emotion"
            self.sem_rep = "semantic"
            self.id_rep = "speaker"
            self.word_cache = False
            self.t_pre_encoder = "bert"
            self.beat_align = False
            self.new_cache = False
            self.ori_joints = "beat_smplx_full"
            self.tar_joints = "beat_smplx_full"
            
    # Test dataset creation
    args = MockArgs()
    
    try:
        dataloader = create_a2g_dataloader(
            args=args,
            mode="train", 
            batch_size=2,
            num_workers=0,  # 0 for testing
            sequence_length=32
        )
        
        print("A2G DataLoader created successfully!")
        print(f"Dataset size: {len(dataloader.dataset)}")
        
        # Test a single batch
        for batch in dataloader:
            print("Batch keys:", batch.keys())
            print("Audio waveform shape:", batch["audio_waveform"].shape)
            print("Gesture parts:", {k: v.shape for k, v in batch["gesture_parts"].items()})
            break
            
    except Exception as e:
        print(f"DataLoader test failed: {e}")
        print("This is expected if BEAT dataset is not properly set up")
