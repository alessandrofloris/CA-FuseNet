import pytest
import os
import numpy as np
import torch
import cv2

from src.dataset.itwpolimi_loader import ITWPOLIMI_Loader 

# ----------------------------------------------------------------------
# FIXTURES AND MOCKING SETUP
# ----------------------------------------------------------------------

# Define key parameters expected by the _load_rgb_tubelet function
T_RGB_TEST = 32
H_W_TEST = 224
FRAME_COUNT_TEST = 60 # Total frames in the dummy video

# MOCK: Function to create a simple dummy video file
def create_dummy_video(path, frame_count, fps=10, width=320, height=240):
    """Creates a temporary MP4 video file for testing purposes."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    # Create a uniform frame (e.g., dark blue color)
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    dummy_frame[:, :, 0] = 100 # Set Blue channel value
    dummy_frame[:, :, 1] = 50  # Set Green channel value
    dummy_frame[:, :, 2] = 20  # Set Red channel value

    for _ in range(frame_count):
        out.write(dummy_frame)
    out.release()

@pytest.fixture(scope="module")
def setup_dummy_loader(tmpdir_factory):
    """
    Creates all necessary dummy files and a mock loader instance for testing __getitem__.
    """
    
    # 1. Create the output directory and video path
    temp_dir = tmpdir_factory.mktemp("data_test")
    dummy_video_path = os.path.join(str(temp_dir), "dummy_video.mp4")
    
    # Create the dummy video file
    create_dummy_video(dummy_video_path, FRAME_COUNT_TEST)
    
    # 2. Mock input data (as if loaded by __init__ for a single sample N=1)
    
    # The action sequence runs from frame 10 to frame 50 (41 raw frames)
    frame_indices_raw = list(range(10, 51)) 
    
    # Bounding Box sequence for the 41 frames (using a fixed central BBox for simplicity)
    # Format: (x1, y1, x2, y2)
    dummy_bbox_seq = np.array([[100, 100, 200, 200]] * len(frame_indices_raw))
    
    # The data loaded in __init__ must be lists/arrays of length 1 (N=1)
    mock_data = {
        'data_ske': np.zeros((1, 2, 300, 17, 1)), # Mock padded pose data
        'data_bbox': np.array([dummy_bbox_seq]), # BBox sequence for the single sample
        'label': torch.LongTensor([0]),
        'video_paths': [dummy_video_path],
        'frame_indices': [frame_indices_raw],
        'root_path': "", 
        'videos_path': ""
    }

    # Create a Mock class to bypass the file loading logic in ITWPOLIMI_Loader.__init__
    class MockITWPOLIMI_Loader(ITWPOLIMI_Loader):
        def __init__(self):
            # Overwrite __init__ to load mock data directly into instance attributes
            self.data_ske = mock_data['data_ske']
            self.data_bbox = mock_data['data_bbox']
            self.label = mock_data['label']
            self.video_paths = mock_data['video_paths']
            self.frame_indices = mock_data['frame_indices']
            self.root_path = mock_data['root_path']
            self.videos_path = mock_data['videos_path']
            self.transform = None 
        
        def __len__(self):
            return 1

    return MockITWPOLIMI_Loader()


# ----------------------------------------------------------------------
# TESTS FOR TUBELET EXTRACTION LOGIC
# ----------------------------------------------------------------------

def test_tubelet_output_shape(setup_dummy_loader):
    """Verifies the final shape of the output RGB tubelet tensor (T_RGB, C, H, W)."""
    loader = setup_dummy_loader
    
    # Call the helper function directly for isolated testing
    rgb_tubelet = loader._load_rgb_tubelet(
        video_path=loader.video_paths[0], 
        frame_indices=loader.frame_indices[0], 
        bbox_data=loader.data_bbox[0], 
        T_RGB=T_RGB_TEST, 
        H_W=H_W_TEST
    )
    
    # Expected shape: (T_RGB, C=3, H_W, H_W)
    assert rgb_tubelet.shape == (T_RGB_TEST, 3, H_W_TEST, H_W_TEST)
    assert rgb_tubelet.dtype == torch.float32


def test_spatial_cropping_and_normalization(setup_dummy_loader):
    """Verifies pixel values are normalized [0, 1] and color channel conversion is correct."""
    loader = setup_dummy_loader
    
    rgb_tubelet = loader._load_rgb_tubelet(
        video_path=loader.video_paths[0], 
        frame_indices=loader.frame_indices[0], 
        bbox_data=loader.data_bbox[0], 
        T_RGB=T_RGB_TEST, 
        H_W=H_W_TEST
    )
    
    # 1. Verifies pixel values are within the normalized range [0, 1]
    assert torch.all(rgb_tubelet >= 0.0)
    assert torch.all(rgb_tubelet <= 1.0)
    
    # 2. Verifies color channel mapping (BGR -> RGB in the loader)
    # Dummy video color (BGR): B=100, G=50, R=20. Normalized (0-1): B=0.39, G=0.19, R=0.07
    
    # Check the Blue channel mean (should be at index 2 in RGB output)
    mean_blue_channel = torch.mean(rgb_tubelet[:, 2, :, :]).item() 
    assert 0.3 < mean_blue_channel < 0.45 
    
    # Check the Red channel mean (should be at index 0 in RGB output)
    mean_red_channel = torch.mean(rgb_tubelet[:, 0, :, :]).item() 
    assert 0.05 < mean_red_channel < 0.15 # Around 20/255 = 0.078
    
    
def test_integration_getitem(setup_dummy_loader):
    """Verifies that the full __getitem__ method returns the expected dictionary structure."""
    loader = setup_dummy_loader
    
    # Temporarily update __getitem__ logic to return the complete planned dictionary
    # (assuming missing crowd_awareness logic is currently commented or mocked)
    
    sample = loader[0]
    
    # Verify the dictionary keys
    assert 'video' in sample
    assert 'pose' in sample
    assert 'label' in sample
    
    # Verify the final output shapes
    assert sample['video'].shape == (T_RGB_TEST, 3, H_W_TEST, H_W_TEST)
    assert sample['pose'].shape == (2, 300, 17, 1) 
    assert sample['label'].ndim == 0 or sample['label'].ndim == 1 # Scalar or 1D tensor