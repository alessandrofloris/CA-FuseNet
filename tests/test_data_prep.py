import pytest
import os
import pandas as pd
from src.dataset.itwpolimi_loader import ITWPOLIMI_Loader 

# ----------------------------------------------------------------------
# INITIAL TEST CONFIGURATION
# ----------------------------------------------------------------------
TEST_ROOT_DIR = 'D:/Dateset_mock_POLIMI-ITW-S'
TEST_VIDEOS_DIR = 'blurred_RGB_video'
TEST_ANNOTATIONS_DIR = 'annotation_skeleton_bbox_label'

# Expected number of samples in the subset
EXPECTED_SAMPLES_COUNT = 11
# (Example: if you have 5 'cleaning' videos and 6 'jumping' videos with corresponding JSONs)
# ----------------------------------------------------------------------

# Initializes the Dataset once
@pytest.fixture(scope="module")
def dataset_instance():
    """Returns an instance of the ITWPOLIMI_Loader for all module tests."""
    if not os.path.exists(TEST_ROOT_DIR):
        pytest.skip(f"The test root directory does not exist: {TEST_ROOT_DIR}")
    return ITWPOLIMI_Loader(root_path=TEST_ROOT_DIR, annotations_path=TEST_ANNOTATIONS_DIR, videos_path=TEST_VIDEOS_DIR)

# ----------------------------------------------------------------------
# __init__ and __len__ FUNCTIONALITY TESTS
# ----------------------------------------------------------------------

def test_initialization_success(dataset_instance):
    """Verifies that initialization does not raise exceptions."""
    # If the fixture executed successfully, initialization is OK
    assert dataset_instance is not None
    # Verifies that self.samples is created as a DataFrame
    assert isinstance(dataset_instance.samples, pd.DataFrame)
    
def test_correct_sample_count(dataset_instance):
    """Verifies that __len__ returns the correct number of samples."""
    # Checks that the index size matches the expected count
    assert len(dataset_instance) == EXPECTED_SAMPLES_COUNT

def test_sample_structure(dataset_instance):
    """Verifies that the index contains the essential columns (path and category)."""
    required_columns = ['video_path', 'json_path', 'category']
    
    # Checks that all required columns are present in the DataFrame
    assert all(col in dataset_instance.samples.columns for col in required_columns)
    
def test_file_paths_exist(dataset_instance):
    """Verifies that all paths in the index point to existing files on disk."""
    
    samples_df = dataset_instance.samples
    
    # Iterates over the index and checks the existence of both files
    for _, row in samples_df.iterrows():
        video_exists = os.path.exists(os.path.join(TEST_ROOT_DIR, row['video_path']))
        json_exists = os.path.exists(os.path.join(TEST_ROOT_DIR, row['json_path']))
        
        # Fails the test if either path does not exist
        if not video_exists or not json_exists:
            pytest.fail(
                f"Missing file error. Video Path: {os.path.join(TEST_ROOT_DIR, row['video_path'])} (Exists: {video_exists}), "
                f"JSON Path: {os.path.join(TEST_ROOT_DIR, row['json_path'])} (Exists: {json_exists})"
            )

def test_path_format_and_association(dataset_instance):
    """Verifies the file name consistency (e.g., 'b_' prefix and VID_ID association)."""
    samples_df = dataset_instance.samples
    
    for index, row in samples_df.iterrows():
        video_name = os.path.basename(row['video_path'])
        json_name = os.path.basename(row['json_path'])
        
        # 1. Check for the 'b_' prefix in the video name
        assert video_name.startswith('b_'), f"Video name does not start with 'b_': {video_name}"
        
        # 2. Verify that the base names of the files match (excluding prefixes)
        # Example: 'b_VID_123.mp4' must correspond to 'action_VID_123.json'
        
        # Extract the video ID from the JSON name (after 'action_' and before '.json')
        expected_vid_id = json_name.replace('action_', '').replace('.json', '')
        
        # Check that the extracted video ID is contained in the video file name
        assert expected_vid_id in video_name, f"Incorrect association for sample {index}. JSON ID: {expected_vid_id}, Video Name: {video_name}"