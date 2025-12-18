import yaml, argparse, os
from time import sleep
import logging
from processor.processor import Processor

from utils.logger import setup_logger

def init_parser():
    parser = argparse.ArgumentParser(description='Method for Skeleton-based Action Recognition')

    # --- Settings ---
    parser.add_argument('--config', '-c', type=str, default='default', help='Name of the using config')
    parser.add_argument('--log_folder', type=str, default='../logs', help='Log folder path')
    parser.add_argument('--log_name', type=str, default='log', help='Name of the log file')

    # --- Paths ---
    parser.add_argument('--root_path', type=str, default='C:/Users/flori/Documents/MECIN/CA-FuseNet/', help='Project root directory')
    parser.add_argument('--videos_path', type=str, default='blurred_RGB_video/', help='Subfolder containing the raw videos')
    parser.add_argument('--data_path', type=str, default='data_preprocessed/', help='Subfolder containing pre-processed data')

    # --- Processing ---
    parser.add_argument('--extract', '-ex', default=False, action='store_true', help='Extract')

    # --- Model ---
    parser.add_argument('--pose_channels', type=int, default=2, help='Number of channels for pose data (e.g., x, y coordinates)')
    parser.add_argument('--pose_joints', type=int, default=17, help='Number of joints in the pose data')
    parser.add_argument('--pose_length', type=int, default=300, help='Maximum length (T) of the pose sequences')

    # --- Embedding and Fusion Dimensions ---
    parser.add_argument('--pose_embedding_dim', type=int, default=256, help='Dimension of the embedding for the pose modality')
    parser.add_argument('--video_embedding_dim', type=int, default=256, help='Dimension of the embedding for the video modality')
    parser.add_argument('--fused_embedding_dim', type=int, default=512, help='Dimension of the fused embedding (pose + video)')

    # --- Classification ---
    parser.add_argument('--num_classes', type=int, default=15, help='Number of action classes for classification')

    # --- DataLoader Parameters ---
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')

    # --- Training ---
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation (e.g., "cuda" or "cpu")')

    # --- Video Tubelet ---
    parser.add_argument('--t_rgb', type=int, default=32, help='Length of the sampled video tubelet (T_RGB)')
    parser.add_argument('--h_w', type=int, default=224, help='Height and Width of the cropped video frames (H x W)')

    return parser


def update_parameters(parser, args):
    if os.path.exists('../configs/{}.yaml'.format(args.config)):
        with open('../configs/{}.yaml'.format(args.config), 'r') as f:
            try:
                yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_arg = yaml.load(f)
            default_arg = vars(args)
            for k in yaml_arg.keys():
                if k not in default_arg.keys():
                    raise ValueError('Do NOT exist this parameter {}'.format(k))
            parser.set_defaults(**yaml_arg)
    else:
        raise ValueError('Do NOT exist this file in \'configs\' folder: {}.yaml!'.format(args.config))
    return parser.parse_args()

if __name__ == '__main__':
    
    
    # Loading parameters
    parser = init_parser()
    args = parser.parse_args()
    args = update_parameters(parser, args)  # Override priority cmd > yaml > default

    # Logger setup
    logger = setup_logger(
            log_dir=f"{args.log_folder}",
            log_filename=f"{args.log_name}_run.log",
            level=logging.INFO
    )
    logger.info("Configuration logging initialized.")

    # Processing
    if args.extract:
        if args.extract:
            # Post training for inspection and plotting
            p = Processor(args)
            p.extract()
    else:
        # Training or Evaluation
        p = Processor(args)
        p.start()
    