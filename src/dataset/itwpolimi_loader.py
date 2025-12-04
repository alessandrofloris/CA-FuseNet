import os
import pandas as pd
from torch.utils.data import Dataset

class ITWPOLIMI_Loader(Dataset):

    def __init__(self, root_path, annotations_path, videos_path, transform=None):
        '''
        This method initializes the dataset loader with the given parameters.
        '''
        self.root_path = root_path
        self.annotations_path = annotations_path
        self.videos_path = videos_path
        self.transform = transform

        self.samples = self.__create_sample_list()


    def __len__(self):
        '''
        This method should return the total number of samples in the dataset.
        '''
        return len(self.samples)


    def __getitem__(self, idx):
        '''
        This method should load and return a sample from the dataset at the given index.

        A sample is a tuple consisting of a video, pose data, bounding boxes, and activity labels.
        Everything should be properly preprocessed and transformerd in a tensor format.
        '''

        category = self.samples[idx]['category']
        video_path = os.path.join(self.root_path, self.samples[idx]['video_path'])
        json_path = os.path.join(self.root_path, self.samples[idx]['json_path'])
        
        # video = load_video(video_path)
        # pose_data, bboxes, labels = load_and_parse_annotations(json_path)


    def __create_sample_list(self):
        '''
        This method should create and return a list of samples in the dataset.
        Each sample is represented as a dictionary with keys: video_path, json_path, category.

        example: samples[i] = {'video_path': 'path/to/video.mp4', 'json_path': 'path/to/annotation, category': 'some_category'}
        '''
        samples = []
        
        # Logic to populate the samples list by scanning the annotations and videos directories        

        annotations_path = os.path.join(self.root_path, self.annotations_path)
        for category in os.listdir(annotations_path):
            category_path = os.path.join(annotations_path, category)
            if not os.path.isdir(category_path):
                continue
            
            for json_file in os.listdir(category_path):
                if not json_file.endswith('.json'):
                    continue
                
                json_path = os.path.join(category_path, json_file)
                
                # Derive the corresponding video file name
                base_name = os.path.splitext(json_file)[0]
                base_name = base_name[len('action_'):]

                # Construct the expected video file name (using the 'b_' prefix)
                video_filename = f'b_{base_name}.mp4'
                
                # Construct the full video path
                video_path = os.path.join(self.root_path, self.videos_path, category, video_filename)
                
                # Verify the video file exists before adding the sample
                if os.path.exists(video_path):
                    samples.append({
                        'category': category,
                        'json_path': os.path.relpath(json_path, self.root_path),
                        'video_path': os.path.relpath(video_path, self.root_path)
                    })

        return pd.DataFrame(samples)