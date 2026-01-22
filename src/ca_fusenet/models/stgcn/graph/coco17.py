import numpy as np
from .utils import get_hop_distance, normalize_digraph

NUM_NODES = 17
CENTER = 5  # left_shoulder, otherwise 6 o 11/12

# COCO-17 edges (0-indexed)
NEIGHBOR_LINK = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # head
    (5, 6),                              # shoulders
    (5, 7), (7, 9),                      # left arm
    (6, 8), (8, 10),                     # right arm
    (5, 11), (6, 12), (11, 12),          # torso/hips
    (11, 13), (13, 15),                  # left leg
    (12, 14), (14, 16),                  # right leg
]

SELF_LINK = [(i, i) for i in range(NUM_NODES)]
EDGE = SELF_LINK + NEIGHBOR_LINK

def build_adjacency(strategy="uniform", max_hop=1, dilation=1):
    hop_dis = get_hop_distance(NUM_NODES, EDGE, max_hop=max_hop)
    valid_hop = range(0, max_hop + 1, dilation)

    adjacency = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1

    normalize_adjacency = normalize_digraph(adjacency)

    if strategy == "uniform":
        A = np.zeros((1, NUM_NODES, NUM_NODES), dtype=np.float32)
        A[0] = normalize_adjacency
        return A

    raise ValueError(f"Unsupported strategy: {strategy}")