"""
Real BEAT A2G DataLoader
Based on beat_sep_lower.py with full LMDB caching and multimodal processing
"""

import os
import pickle
import math
import shutil
import numpy as np
import lmdb as lmdb
import textgrid as tg
import pandas as pd
import torch
import glob
import json
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
import pickle
import smplx
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dataloaders.utils.audio_features import process_audio_data
from dataloaders.data_tools import joints_list
from dataloaders.utils.other_tools import MultiLMDBManager
from dataloaders.utils.motion_rep_transfer import process_smplx_motion
from dataloaders.utils.mis_features import process_semantic_data, process_emotion_data
from dataloaders.utils.text_features import process_word_data
from dataloaders.utils.data_sample import sample_from_clip


class BeatA2GRealDataset(Dataset):
    """
    Real BEAT A2G Dataset with full LMDB caching and multimodal processing
    Based on beat_sep_lower.py but optimized for A2G task
    """
    
    def __init__(self, args, loader_type="train", augmentation=None, kwargs=None, build_cache=True):
        self.args = args
        self.loader_type = loader_type
        
        # Initialize distributed training if available
        if dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0

        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        
        # Initialize basic parameters
        self._init_parameters()
        
        # Initialize SMPLX model
        self._init_smplx_model()
        
        # Load and process split rules
        self._process_split_rules()

        # Initialize joint masks
        self._init_joint_masks()
        
        # Initialize data directories and lengths
        self._init_data_paths()

        # Calculate mean velocity if needed for beat alignment
        if hasattr(self.args, 'beat_align') and self.args.beat_align:
            if not os.path.exists(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy"):
                self.calculate_mean_velocity(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
            self.avg_vel = np.load(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
        
        # Build or load cache
        self._init_cache(build_cache)
        
    def _init_parameters(self):
        """Initialize basic parameters for the dataset."""
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        self.alignment = [0, 0]  # for trinity
        
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list = joints_list[self.args.tar_joints]

        if hasattr(self.args, 'word_rep') and self.args.word_rep is not None:
            try:
                with open(f"{self.args.data_path}weights/vocab.pkl", 'rb') as f:
                    self.lang_model = pickle.load(f)
            except (FileNotFoundError, AttributeError, ImportError) as e:
                logger.warning(f"Failed to load vocab.pkl ({e}), using dummy language model")
                self.lang_model = None
        else:
            self.lang_model = None
        
    def _init_joint_masks(self):
        """Initialize joint masks based on pose representation."""
        if 'smplx' in self.args.pose_rep:
            self.joint_mask = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            self.joints = len(list(self.tar_joint_list.keys()))
            for joint_name in self.tar_joint_list:
                if joint_name in self.ori_joint_list:
                    start_idx = self.ori_joint_list[joint_name][0] 
                    end_idx = self.ori_joint_list[joint_name][1]
                    self.joint_mask[start_idx:end_idx] = 1
        else:
            self.joints = len(list(self.ori_joint_list.keys()))+1
            self.joint_mask = np.zeros(self.joints*3)
            for joint_name in self.tar_joint_list:
                if joint_name == "Hips":
                    self.joint_mask[3:6] = 1
                elif joint_name in self.ori_joint_list:
                    start_idx = self.ori_joint_list[joint_name][0]
                    end_idx = self.ori_joint_list[joint_name][1] 
                    self.joint_mask[start_idx:end_idx] = 1
    
    def _init_smplx_model(self):
        """Initialize SMPLX model."""
        smplx_path = getattr(self.args, 'data_path_1', './datasets/hub/') + "smplx_models/"
        if not os.path.exists(smplx_path):
            logger.warning(f"SMPLX models not found at {smplx_path}, using fallback")
            smplx_path = "./datasets/hub/smplx_models/"
            
        try:
            self.smplx = smplx.create(
                smplx_path, 
                model_type='smplx',
                gender='NEUTRAL_2020', 
                use_face_contour=False,
                num_betas=300,
                num_expression_coeffs=100, 
                ext='npz',
                use_pca=False,
            ).cuda().eval()
        except Exception as e:
            logger.warning(f"Failed to initialize SMPLX: {e}")
            self.smplx = None
    
    def _process_split_rules(self):
        """Process dataset split rules."""
        split_file = os.path.join(self.args.data_path, "train_test_split.csv")
        split_rule = pd.read_csv(split_file)
        
        self.selected_file = split_rule.loc[
            (split_rule['type'] == self.loader_type) & 
            (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))
        ]
        
        # Add additional data for training if specified
        if hasattr(self.args, 'additional_data') and self.args.additional_data and self.loader_type == 'train':
            split_b = split_rule.loc[
                (split_rule['type'] == 'additional') & 
                (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))
            ]
            if not split_b.empty:
                self.selected_file = pd.concat([self.selected_file, split_b])
            
        if self.selected_file.empty:
            logger.warning(f"{self.loader_type} is empty for speaker {self.args.training_speakers}, use train set 0-8 instead")
            self.selected_file = split_rule.loc[
                (split_rule['type'] == 'train') & 
                (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))
            ]
            self.selected_file = self.selected_file.iloc[0:8]
            
        logger.info(f"Selected {len(self.selected_file)} files for {self.loader_type} with speakers {self.args.training_speakers}")
    
    def _init_data_paths(self):
        """Initialize data directories and lengths."""
        self.data_dir = self.args.data_path
        
        if self.loader_type == "test":
            if hasattr(self.args, 'multi_length_training'):
                self.args.multi_length_training = [1.0]
            else:
                self.args.multi_length_training = [1.0]
        else:
            if not hasattr(self.args, 'multi_length_training'):
                self.args.multi_length_training = [1.0]
            
        self.max_length = int(self.args.pose_length * self.args.multi_length_training[-1])
        self.max_audio_pre_len = math.floor(self.args.pose_length / self.args.pose_fps * self.args.audio_sr)
        
        test_length = getattr(self.args, 'test_length', 128)
        if self.max_audio_pre_len > test_length * self.args.audio_sr:
            self.max_audio_pre_len = test_length * self.args.audio_sr
        
        # Cache directory
        root_path = getattr(self.args, 'root_path', './')
        cache_path = getattr(self.args, 'cache_path', 'datasets/beat_cache/beat_smplx_en_emage_2_128/')
        self.preloaded_dir = os.path.join(root_path, cache_path, self.loader_type, f"{self.args.pose_rep}_cache")
    
    def _init_cache(self, build_cache):
        """Initialize or build cache."""
        self.lmdb_envs = {}
        self.mapping_data = None
        
        if build_cache and self.rank == 0:
            self.build_cache(self.preloaded_dir)
        
        # Wait for cache to be built in distributed training
        if dist.is_initialized():
            dist.barrier()
        
        self.load_db_mapping()
    
    def build_cache(self, preloaded_dir):
        """Build the dataset cache."""
        logger.info(f"Audio bit rate: {self.args.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        logger.info("Creating the dataset cache...")
        
        new_cache = getattr(self.args, 'new_cache', False)
        if new_cache and os.path.exists(preloaded_dir):
            shutil.rmtree(preloaded_dir)
            
        if os.path.exists(preloaded_dir):
            logger.info("Found the cache {}".format(preloaded_dir))
            return
            
        # Cache generation parameters
        disable_filtering = getattr(self.args, 'disable_filtering', True)
        clean_first_seconds = getattr(self.args, 'clean_first_seconds', 0)
        clean_final_seconds = getattr(self.args, 'clean_final_seconds', 0)
        
        if self.loader_type == "test":
            self.cache_generation(preloaded_dir, True, 0, 0, is_test=True)
        else:
            self.cache_generation(
                preloaded_dir, 
                disable_filtering,
                clean_first_seconds,
                clean_final_seconds,
                is_test=False
            )
    
    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds, clean_final_seconds, is_test=False):
        """Generate cache for the dataset."""
        if not os.path.exists(out_lmdb_dir):
            os.makedirs(out_lmdb_dir)
        
        # Initialize the multi-LMDB manager
        lmdb_manager = MultiLMDBManager(out_lmdb_dir, max_db_size=10*1024*1024*1024)
        
        self.n_out_samples = 0
        n_filtered_out = defaultdict(int)
        
        for index, file_name in self.selected_file.iterrows():
            f_name = file_name["id"]
            ext = ".npz" if "smplx" in self.args.pose_rep else ".bvh"
            pose_file = os.path.join(self.data_dir, self.args.pose_rep, f_name + ext)
            
            if not os.path.exists(pose_file):
                logger.warning(f"Pose file not found: {pose_file}")
                continue
            
            # Process data
            data = self._process_file_data(f_name, pose_file, ext)
            if data is None:
                continue
                
            # Sample from clip
            filtered_result, self.n_out_samples = sample_from_clip(
                lmdb_manager=lmdb_manager,
                audio_file=pose_file.replace(self.args.pose_rep, 'wave16k').replace(ext, ".wav"),
                audio_each_file=data['audio'],
                pose_each_file=data['pose'],
                trans_each_file=data['trans'],
                trans_v_each_file=data['trans_v'],
                shape_each_file=data['shape'],
                facial_each_file=data['facial'],
                word_each_file=data['word'],
                vid_each_file=data['vid'],
                emo_each_file=data['emo'],
                sem_each_file=data['sem'],
                args=self.args,
                ori_stride=self.ori_stride,
                ori_length=self.ori_length,
                disable_filtering=disable_filtering,
                clean_first_seconds=clean_first_seconds,
                clean_final_seconds=clean_final_seconds,
                is_test=is_test,
                n_out_samples=self.n_out_samples
            )
            
            for type_key in filtered_result:
                n_filtered_out[type_key] += filtered_result[type_key]
        
        logger.info(f"Generated {self.n_out_samples} samples for {self.loader_type}")
        for filter_type, count in n_filtered_out.items():
            logger.info(f"Filtered out {count} samples due to {filter_type}")
            
        lmdb_manager.close()
    
    def _process_file_data(self, f_name, pose_file, ext):
        """Process all data for a single file."""
        data = {
            'pose': None, 'trans': None, 'trans_v': None, 'shape': None,
            'audio': None, 'facial': None, 'word': None, 'emo': None,
            'sem': None, 'vid': None
        }
        
        # Process motion data
        logger.info(colored(f"# ---- Building cache for Pose {f_name} ---- #", "blue"))
        if "smplx" in self.args.pose_rep:
            if self.smplx is None:
                logger.error("SMPLX model not initialized")
                return None
            motion_data = process_smplx_motion(
                pose_file, self.smplx, self.joint_mask, 
                self.args.pose_fps, getattr(self.args, 'facial_rep', None)
            )
        else:
            logger.error(f"Unknown pose representation '{self.args.pose_rep}'.")
            return None
            
        if motion_data is None:
            logger.warning(f"Failed to process motion data for {f_name}")
            return None
            
        data.update(motion_data)
        
        # Process speaker ID
        id_rep = getattr(self.args, 'id_rep', None)
        if id_rep is not None:
            try:
                speaker_id = int(f_name.split("_")[0]) - 1
                data['vid'] = np.repeat(np.array(speaker_id).reshape(1, 1), data['pose'].shape[0], axis=0)
            except (IndexError, ValueError):
                logger.warning(f"Could not extract speaker ID from {f_name}")
                data['vid'] = np.array([-1])
        else:
            data['vid'] = np.array([-1])
        
        # Process audio if needed
        audio_rep = getattr(self.args, 'audio_rep', None)
        if audio_rep is not None:
            audio_file = pose_file.replace(self.args.pose_rep, 'wave16k').replace(ext, ".wav")
            if os.path.exists(audio_file):
                data = process_audio_data(audio_file, self.args, data, f_name, self.selected_file)
                if data is None:
                    logger.warning(f"Failed to process audio for {f_name}")
                    return None
            else:
                logger.warning(f"Audio file not found: {audio_file}")
                # Create dummy audio data
                audio_len = int(data['pose'].shape[0] / self.args.pose_fps * self.args.audio_sr)
                data['audio'] = np.zeros((audio_len, 2))  # Assuming stereo audio
        
        # Process emotion if needed - Use dummy data for now
        data['emo'] = np.zeros((data['pose'].shape[0], 1))
        
        # emo_rep = getattr(self.args, 'emo_rep', None)
        # if emo_rep is not None:
        #     try:
        #         data = process_emotion_data(f_name, data, self.args)
        #     except Exception as e:
        #         logger.warning(f"Failed to process emotion for {f_name}: {e}")
        #         data['emo'] = np.zeros((data['pose'].shape[0], 1))
        
        # Process word data if needed - Use dummy data for now
        word_dims = getattr(self.args, 'word_dims', 300)
        data['word'] = np.zeros((data['pose'].shape[0], word_dims))
        
        # word_rep = getattr(self.args, 'word_rep', None)
        # if word_rep is not None and self.lang_model is not None:
        #     word_file = f"{self.data_dir}{word_rep}/{f_name}.TextGrid"
        #     if os.path.exists(word_file):
        #         try:
        #             data = process_word_data(self.data_dir, word_file, self.args, data, f_name, self.selected_file, self.lang_model)
        #         except Exception as e:
        #             logger.warning(f"Failed to process word data for {f_name}: {e}")
        #             data['word'] = np.zeros((data['pose'].shape[0], word_dims))
        
        # Process semantic data if needed - Use dummy data for now
        data['sem'] = np.zeros((data['pose'].shape[0], 1))
        
        # sem_rep = getattr(self.args, 'sem_rep', None)
        # if sem_rep is not None:
        #     sem_file = f"{self.data_dir}{sem_rep}/{f_name}.txt"
        #     if os.path.exists(sem_file):
        #         try:
        #             data = process_semantic_data(sem_file, self.args, data, f_name)
        #         except Exception as e:
        #             logger.warning(f"Failed to process semantic data for {f_name}: {e}")
        #             data['sem'] = np.zeros((data['pose'].shape[0], 1))
        
        return data
        
    def load_db_mapping(self):
        """Load database mapping from file."""
        mapping_path = os.path.join(self.preloaded_dir, "sample_db_mapping.pkl")
        
        if not os.path.exists(mapping_path):
            logger.error(f"Mapping file not found: {mapping_path}")
            logger.error("Please ensure cache has been built successfully")
            raise FileNotFoundError(f"Cache mapping not found: {mapping_path}")
            
        try:
            with open(mapping_path, 'rb') as f:
                self.mapping_data = pickle.load(f)
            self.n_samples = len(self.mapping_data['mapping'])
            logger.info(f"Loaded {self.n_samples} samples from cache")
        except Exception as e:
            logger.error(f"Failed to load mapping data: {e}")
            raise
    
    def get_lmdb_env(self, db_idx):
        """Get LMDB environment for given database index."""
        if db_idx not in self.lmdb_envs:
            if db_idx not in self.mapping_data['db_paths']:
                logger.error(f"Database index {db_idx} not found in mapping")
                raise KeyError(f"DB index {db_idx} not found")
            
            db_path = self.mapping_data['db_paths'][db_idx]
            if not os.path.exists(db_path):
                logger.error(f"Database path not found: {db_path}")
                raise FileNotFoundError(f"DB path not found: {db_path}")
                
            self.lmdb_envs[db_idx] = lmdb.open(db_path, readonly=True, lock=False)
        return self.lmdb_envs[db_idx]
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.n_samples
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.n_samples}")
            
        try:
            db_idx = self.mapping_data['mapping'][idx]
            lmdb_env = self.get_lmdb_env(db_idx)
            
            with lmdb_env.begin(write=False) as txn:
                key = "{:008d}".format(idx).encode("ascii")
                sample = txn.get(key)
                
                if sample is None:
                    logger.warning(f"Sample {idx} not found in LMDB")
                    return self._get_dummy_sample()
                
                sample = pickle.loads(sample)
                
                # Unpack sample data
                tar_pose, in_audio, in_facial, in_shape, in_word, emo, sem, vid, trans, trans_v, audio_name = sample
                
                # Convert data to tensors with appropriate types
                processed_data = self._convert_to_tensors(
                    tar_pose, in_audio, in_facial, in_shape, in_word,
                    emo, sem, vid, trans, trans_v
                )
                
                processed_data['audio_name'] = audio_name
                return processed_data
                
        except Exception as e:
            logger.warning(f"Failed to load sample {idx}: {e}")
            return self._get_dummy_sample()
    
    def _convert_to_tensors(self, tar_pose, in_audio, in_facial, in_shape, in_word,
                           emo, sem, vid, trans, trans_v):
        """Convert numpy arrays to tensors with appropriate types."""
        word_cache = getattr(self.args, 'word_cache', False)
        
        data = {
            'emo': torch.from_numpy(emo).int() if emo is not None else torch.zeros(1, dtype=torch.int),
            'sem': torch.from_numpy(sem).float() if sem is not None else torch.zeros(1, dtype=torch.float),
            'audio': torch.from_numpy(in_audio).float() if in_audio is not None else torch.zeros(1, dtype=torch.float),
            'word': torch.from_numpy(in_word).float() if word_cache else torch.from_numpy(in_word).int() if in_word is not None else torch.zeros(1, dtype=torch.int)
        }
        
        if self.loader_type == "test":
            data.update({
                'pose': torch.from_numpy(tar_pose).float() if tar_pose is not None else torch.zeros(1, dtype=torch.float),
                'trans': torch.from_numpy(trans).float() if trans is not None else torch.zeros(1, dtype=torch.float),
                'trans_v': torch.from_numpy(trans_v).float() if trans_v is not None else torch.zeros(1, dtype=torch.float),
                'facial': torch.from_numpy(in_facial).float() if in_facial is not None else torch.zeros(1, dtype=torch.float),
                'id': torch.from_numpy(vid).float() if vid is not None else torch.zeros(1, dtype=torch.float),
                'beta': torch.from_numpy(in_shape).float() if in_shape is not None else torch.zeros(1, dtype=torch.float)
            })
        else:
            data.update({
                'pose': torch.from_numpy(tar_pose).reshape((tar_pose.shape[0], -1)).float() if tar_pose is not None else torch.zeros(1, dtype=torch.float),
                'trans': torch.from_numpy(trans).reshape((trans.shape[0], -1)).float() if trans is not None else torch.zeros(1, dtype=torch.float),
                'trans_v': torch.from_numpy(trans_v).reshape((trans_v.shape[0], -1)).float() if trans_v is not None else torch.zeros(1, dtype=torch.float),
                'facial': torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float() if in_facial is not None else torch.zeros(1, dtype=torch.float),
                'id': torch.from_numpy(vid).reshape((vid.shape[0], -1)).float() if vid is not None else torch.zeros(1, dtype=torch.float),
                'beta': torch.from_numpy(in_shape).reshape((in_shape.shape[0], -1)).float() if in_shape is not None else torch.zeros(1, dtype=torch.float)
            })
        
        return data
    
    def _get_dummy_sample(self):
        """Return a dummy sample for error cases."""
        seq_len = getattr(self.args, 'pose_length', 128)
        
        if self.loader_type == "test":
            return {
                'pose': torch.randn(seq_len, 330),  # Standard pose dims
                'trans': torch.randn(seq_len, 3),
                'trans_v': torch.randn(seq_len, 3),
                'facial': torch.randn(seq_len, 100),
                'audio': torch.randn(seq_len * 16, 2),  # 16 audio samples per pose frame
                'word': torch.zeros(seq_len, 300, dtype=torch.float),
                'emo': torch.zeros(seq_len, dtype=torch.int),
                'sem': torch.zeros(seq_len, dtype=torch.float),
                'id': torch.zeros(seq_len, dtype=torch.float),
                'beta': torch.randn(seq_len, 300),
                'audio_name': 'dummy'
            }
        else:
            return {
                'pose': torch.randn(seq_len, 330),
                'trans': torch.randn(seq_len, 3),
                'trans_v': torch.randn(seq_len, 3),
                'facial': torch.randn(seq_len, 100),
                'audio': torch.randn(seq_len * 16, 2),
                'word': torch.zeros(seq_len, 300, dtype=torch.float),
                'emo': torch.zeros(seq_len, dtype=torch.int),
                'sem': torch.zeros(seq_len, dtype=torch.float),
                'id': torch.zeros(seq_len, 1, dtype=torch.float),
                'beta': torch.randn(seq_len, 300),
                'audio_name': 'dummy'
            }


def create_beat_a2g_real_dataloader(args, loader_type="train", build_cache=True, **kwargs):
    """
    Create real BEAT A2G dataloader with LMDB caching
    
    Args:
        args: Configuration object with dataset parameters
        loader_type: "train", "val", or "test"
        build_cache: Whether to build LMDB cache if not exists
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        torch.utils.data.DataLoader: Configured data loader
    """
    dataset = BeatA2GRealDataset(
        args=args,
        loader_type=loader_type,
        build_cache=build_cache
    )
    
    # DataLoader parameters
    batch_size = kwargs.get('batch_size', getattr(args, 'batch_size', 4))
    shuffle = kwargs.get('shuffle', loader_type == "train")
    num_workers = kwargs.get('num_workers', 4)
    drop_last = kwargs.get('drop_last', loader_type == "train")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    
    logger.info(f"Created {loader_type} dataloader: {len(dataset)} samples, {len(dataloader)} batches")
    return dataloader


class DataArgs:
    """
    Configuration class for BEAT A2G dataset based on shortcut_rvqvae_128.yaml
    """
    def __init__(self):
        # Basic paths
        self.root_path = "./"
        self.data_path = "./datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/"
        self.data_path_1 = "./datasets/hub/"
        self.cache_path = "datasets/beat_cache/beat_smplx_en_emage_2_128/"
        
        # Training configuration
        self.training_speakers = [2, 3]  # Start with 2 speakers
        self.additional_data = False
        self.new_cache = False
        
        # Joint and pose configuration
        self.ori_joints = "beat_smplx_joints"
        self.tar_joints = "beat_smplx_full"
        self.pose_rep = "smplxflame_30"
        self.pose_fps = 30
        self.pose_dims = 330
        self.pose_length = 128
        self.stride = 20
        self.test_length = 128
        self.rot6d = True
        self.pre_frames = 4
        self.m_fix_pre = False
        
        # Audio configuration
        self.audio_rep = "onset+amplitude"
        self.audio_sr = 16000
        self.audio_fps = 16000
        self.audio_norm = False
        self.audio_f = 256
        self.audio_raw = None
        
        # Text configuration
        self.word_rep = "textgrid"
        self.word_dims = 300
        self.word_cache = False
        self.t_pre_encoder = "fasttext"
        
        # Facial configuration
        self.facial_rep = "smplxflame_30"
        self.facial_dims = 100
        self.facial_norm = False
        self.facial_f = 0
        
        # Identity configuration
        self.id_rep = "onehot"
        self.speaker_f = 0
        
        # Misc configuration
        self.emo_rep = None
        self.sem_rep = None
        self.beat_align = False
        
        # Training parameters
        self.batch_size = 4
        self.multi_length_training = [1.0]
        self.disable_filtering = True
        self.clean_first_seconds = 0
        self.clean_final_seconds = 0


if __name__ == "__main__":
    """Test the real dataloader"""
    # Create configuration
    args = DataArgs()
    
    logger.info("🧪 Testing BEAT A2G Real DataLoader...")
    
    # Test dataset creation
    try:
        train_loader = create_beat_a2g_real_dataloader(
            args=args,
            loader_type="train",
            build_cache=True,
            batch_size=2,
            num_workers=0  # Single process for testing
        )
        
        logger.info(f"✅ Train loader created: {len(train_loader)} batches")
        
        # Test loading a batch
        for batch_idx, batch in enumerate(train_loader):
            logger.info(f"Batch {batch_idx}: {batch['pose'].shape}")
            logger.info(f"Audio shape: {batch['audio'].shape}")
            logger.info(f"Facial shape: {batch['facial'].shape}")
            logger.info(f"Trans shape: {batch['trans'].shape}")
            logger.info(f"Files: {batch['audio_name']}")
            
            if batch_idx >= 2:  # Test first 3 batches
                break
                
        logger.info("🎉 Real dataloader test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
