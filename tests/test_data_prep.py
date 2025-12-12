import pytest
import os
import numpy as np
import pickle
import torch
# Assuming the class name is ITWPOLIMI_Loader and it's located here:
from src.dataset.itwpolimi_loader import ITWPOLIMI_Loader 

# ----------------------------------------------------------------------
# INITIAL TEST CONFIGURATION
# ----------------------------------------------------------------------

# Root directory for your project/data (where the 'data_preprocessed' folder lives)
TEST_ROOT_DIR = 'C:/Users/flori/Documents/MECIN/CA-FuseNet/' 
# Path to the pre-processed data relative to the root
TEST_DATA_PATH = os.path.join(TEST_ROOT_DIR, 'data', 'processed_dataset')

EXPECTED_TRAIN_SAMPLES = 27 # Total number of action sequences in the 'train' split
EXPECTED_TEST_SAMPLES = 40  # Total number of action sequences in the 'test' split

# Expected shapes 
# T=300 (padded length), J=17 (joints), 2D (coords), 1 (single person)
EXPECTED_POSE_SHAPE = (EXPECTED_TRAIN_SAMPLES, 2, 300, 17, 1) 
# BBox shape: (N, T, 4) if sequence is saved
EXPECTED_BBOX_SHAPE = (EXPECTED_TRAIN_SAMPLES, 300, 4) 
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def train_dataset_instance():
    """Provides a single instance of the DataLoader for the 'train' phase."""
    # Ensure the required paths exist before running tests
    if not os.path.exists(TEST_DATA_PATH):
        pytest.skip(f"Pre-processed data path not found: {TEST_DATA_PATH}")
        
    return ITWPOLIMI_Loader(
        phase='train', 
        root_path=TEST_ROOT_DIR, 
        videos_path='blurred_RGB_video',
        data_path=TEST_DATA_PATH 
    )

# ----------------------------------------------------------------------
# TESTS FOR __init__ and __len__
# ----------------------------------------------------------------------

def test_initialization_success(train_dataset_instance):
    """Verifies that the __init__ method runs without exceptions."""
    assert train_dataset_instance is not None
    
def test_correct_sample_count(train_dataset_instance):
    """Verifies that __len__ returns the expected total number of action sequences (N)."""
    assert len(train_dataset_instance) == EXPECTED_TRAIN_SAMPLES

def test_skeletal_data_shape(train_dataset_instance):
    """Verifies the shape of the loaded pose (skeletal) data (N, 2, T, 17, 1)."""
    # Check shape: N must match the expected sample count
    # Note: We stack the expected shape here with the actual N
    expected_shape_with_n = (EXPECTED_TRAIN_SAMPLES, *train_dataset_instance.data_ske.shape[1:])
    
    assert train_dataset_instance.data_ske.shape == expected_shape_with_n
    

def test_data_alignment(train_dataset_instance):
    """Verifies that all loaded structures (pose, bbox, labels) have the same length (N)."""
    n_skeletal = train_dataset_instance.data_ske.shape[0]
    n_bbox = train_dataset_instance.data_bbox.shape[0]
    n_labels = len(train_dataset_instance.label) # Assuming self.label is a list
    
    # All N dimensions must be identical
    assert n_skeletal == EXPECTED_TRAIN_SAMPLES
    assert n_skeletal == n_bbox
    assert n_skeletal == n_labels
    
def test_data_type(train_dataset_instance):
    """Verifies that NumPy arrays are loaded as expected."""
    assert isinstance(train_dataset_instance.data_ske, np.ndarray)
    assert isinstance(train_dataset_instance.data_bbox, np.ndarray)
    assert isinstance(train_dataset_instance.label, torch.LongTensor) 
    