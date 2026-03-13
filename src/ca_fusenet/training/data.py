import hydra 
import torch
from ca_fusenet.utils.engine import build_dataloader
from ca_fusenet.data.transforms import TransformSubset, VideoTransform, PoseTransform

def buildDataloaders(seed, data_config, training_config):
    
    data_cfg = data_config.get("train", data_config)

    # Train dataset instantiation
    train_dataset = hydra.utils.instantiate(data_cfg)
    if len(train_dataset) <= 0:
        raise ValueError("Training dataset is empty; check data configuration and artifacts.")

    # Get all labels for future class weight calculation
    all_labels = train_dataset.train.get_labels()
    
    # Transformations setup
    video_augmentation = training_config.get("video_augmentation", False)
    pose_augmentation = training_config.get("pose_augmentation", False)
    if video_augmentation:
        crop_size = training_config.get("crop_size", 112)
        train_transform = VideoTransform(train=True, crop_size=crop_size)
        val_transform = VideoTransform(train=False, crop_size=crop_size)
    elif pose_augmentation:
        train_transform = PoseTransform(train=True)
        val_transform = PoseTransform(train=False)
    else:
        train_transform = None
        val_transform = None
    

    # Train/Val split
    val_split = training_config.get("val_split", 0.1)

    if val_split == 0.0:
        train_ds = TransformSubset(
            train_dataset,
            list(range(len(train_dataset))),
            transform=train_transform,
        )
        val_ds = None
    else:
        n_samples = len(train_dataset)
        generator = torch.Generator().manual_seed(seed)
        val_size = max(1, int(n_samples * val_split))
        if val_size >= n_samples:
            val_size = n_samples - 1
        train_size = n_samples - val_size
        indices = torch.randperm(n_samples, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_ds = TransformSubset(train_dataset, train_indices, transform=train_transform)
        val_ds = TransformSubset(train_dataset, val_indices, transform=val_transform)    

    # Data loaders instantiation
    batch_size = training_config.get("batch_size", 32)
    num_workers = training_config.get("num_workers", 4)
    pin_memory = torch.cuda.is_available()

    train_loader = build_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = build_dataloader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    return train_loader, val_loader, all_labels