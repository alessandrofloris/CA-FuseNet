from collections import Counter
import numpy as np
import torch
from torch import nn
import math    
import logging

logger = logging.getLogger(__name__)


def buildCriterion(all_labels, device):
    
    logger.info("Calculating class weights for weighted CrossEntropyLoss...")
        
    counts = Counter(all_labels)
    class_indices = sorted(counts.keys())
    num_samples_per_class = [counts[i] for i in class_indices]
    
    total_samples = sum(num_samples_per_class)
    num_classes = len(num_samples_per_class)
    
    #weights = [math.sqrt(total_samples / (num_classes * count)) for count in num_samples_per_class] # sqrt
    weights = [math.log(1 + total_samples / (num_classes * count)) for count in num_samples_per_class] # log
    class_weights = torch.FloatTensor(weights).to(device)
    
    logger.info("Class weights: %s", weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    return criterion