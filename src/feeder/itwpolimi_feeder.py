import os
import pandas as pd
import numpy as np
import pickle
import cv2
import torch
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class ITWPOLIMI_Feeder(Dataset):

    def __init__(self, phase, root_path, videos_path, data_path, transform=None):
        '''
        This method initializes the dataset feeder with the given parameters.

        Here i load the .pkl files and the .npy files from the specified paths,
        specifically:
        - {phase}_data_joint.npy that contains the pose data
        - {phase}_data_bbox.npy that contains the bounding boxes
        - {phase}_label.pkl that contains the activity labels and other metadata

        Parameters:
        - phase: str, either 'train' or 'test' to specify the dataset split
        - root_path: str, the root directory of the dataset
        - videos_path: str, the path to the videos directory relative to root_path
        - data_path: str, the path to the .npy and .pkl files

        {phase}_label.pkl contains:
        - sample_name: list of str, names of the samples
        - label: list of int, activity labels for each sample
        - frame_indices: list of list of int, frame indices for each sample
        - video_paths: list of str, paths to the original videos for each sample
        '''
        self.root_path = root_path
        self.videos_path = videos_path
        self.data_path = data_path
        self.transform = transform
        
        data_ske_path = '{}/{}_data_joint.npy'.format(data_path, phase)
        data_bbox_path = '{}/{}_data_bbox.npy'.format(data_path, phase)
        labels_path = '{}/{}_label.pkl'.format(data_path, phase)

        # Load the skeletal data
        self.data_ske = np.load(data_ske_path)
        
        # Load bounding box data
        self.data_bbox = np.load(data_bbox_path)
        
        # Load labels and metadata from the pickle file
        with open(labels_path, 'rb') as f:
            self.sample_name, self.label, self.frame_indices, self.video_paths = pickle.load(f)
        
        # Convert labels to tensor
        self.label = torch.LongTensor(self.label)

    def __len__(self):
        '''
        This method returns the total number of samples in the dataset.
        '''
        return len(self.label)


    def __getitem__(self, idx):
        '''
        This method loads and return a sample from the dataset at the given index.
        '''
        # 1. Load Pre-processed Data

        # Skeletal data (already normalized and padded, e.g., (2, 300, 17, 1))
        ske_data = self.data_ske[idx] 
        ske_data = torch.from_numpy(ske_data).float() 

        # Remove last dimension (person id)
        ske_data = ske_data.squeeze(-1)  

        # Permutation needed for the model: (C, T, V) => (T, V=J, C)
        # TODO: Maybe i can avoid this permutation if i change the PyHAPT code accordingly
        ske_data = ske_data.permute(1, 2, 0)

        # Load bounding box data
        bbox_data = self.data_bbox[idx]

        # Load labels and metadata
        label = self.label[idx]
        video_path = os.path.join(self.root_path, self.videos_path, self.video_paths[idx])
        frame_indices = self.frame_indices[idx] 
        
        # 2. RGB Tubelet Extraction 
        # We need to define T_rgb (e.g., 16 or 32) and the target resolution (e.g., 224x224)
        T_RGB = 32
        H_W = 224

        # This helper function performs temporal sampling, spatial cropping, and normalization.
        rgb_tubelet = self._load_rgb_tubelet(video_path, frame_indices, bbox_data, T_RGB, H_W)

        return {
            'video': rgb_tubelet,    # (T_RGB, C, H, W)
            'pose': ske_data,       # (T, J, C)
            'label': label
        }
    
    def _load_rgb_tubelet(self, video_path, frame_indices, bbox_data, T_RGB, H_W):
        '''
        Extracts a fixed-length RGB tubelet by performing temporal sampling and spatial cropping 
        around the bounding box of the tracked person.
        
        Returns: A torch tensor of shape (T_RGB, 3, H_W, H_W).
        '''
        # 1. Temporal Sampling: Select T_RGB frames indices
        total_raw_frames = len(frame_indices)
        
        # Calculate indices to sample T_RGB frames uniformly
        sampling_indices_raw = np.linspace(0, total_raw_frames - 1, T_RGB, dtype=int)
        
        # Get the actual frame indices from the video path to be loaded
        frames_to_load = np.array(frame_indices)[sampling_indices_raw]
        
        # Sync BBoxes: Select the corresponding BBoxes for the sampled frames
        sampled_bboxes = bbox_data[sampling_indices_raw]
        
        # 2. Video Decoding and Tubelet Cropping
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        tubelet_list = []
        
        # The frame index counter in the video file
        frame_counter = 0 
        
        for frame_idx_in_video, bbox in zip(frames_to_load, sampled_bboxes):
            
            # Skip frames until we reach the desired frame_idx_in_video
            while frame_counter < frame_idx_in_video:
                cap.read()
                frame_counter += 1
            
            # Read the target frame
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx_in_video} from {video_path}. Using black frame.")
                # Use a black frame if frame read fails
                frame = np.zeros((H_W, H_W, 3), dtype=np.uint8) 

            frame_counter += 1 # Frame read

            # 3. Spatial Cropping (Tubelet) with Padding
            
            # BBox coordinates are assumed to be (x1, y1, x2, y2)
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Calculate video dimensions
            H_video, W_video, _ = frame.shape

            # Padding logic
            padding_x = int((x2 - x1) * 0.1)
            padding_y = int((y2 - y1) * 0.1)
            
            # Apply padding and ensure coordinates stay within video bounds
            x1_crop = max(0, x1 - padding_x)
            y1_crop = max(0, y1 - padding_y)
            x2_crop = min(W_video, x2 + padding_x)
            y2_crop = min(H_video, y2 + padding_y)
            
            # Crop the frame
            cropped_frame = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            
            # Resize to target resolution (H_W x H_W)
            resized_frame = cv2.resize(cropped_frame, (H_W, H_W))
            
            # 4. Normalization and Format Conversion
            
            # Convert BGR (OpenCV default) to RGB, then to float [0, 1]
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # PyTorch format: (H, W, C) -> (C, H, W)
            rgb_frame = np.transpose(rgb_frame, (2, 0, 1))
            
            tubelet_list.append(rgb_frame)

        cap.release()
        
        # Stack all frames: (T_RGB, C, H, W)
        rgb_tubelet = torch.as_tensor(np.stack(tubelet_list, axis=0)).float()
        
        # Apply torchvision mean/std normalization using self.transform
        if self.transform is not None:
             rgb_tubelet = self.transform(rgb_tubelet)
             
        return rgb_tubelet
