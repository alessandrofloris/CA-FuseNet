# 🚀 CA-FuseNet: Crowd-Aware Fusion Network for Composite Human Activity Recognition

This repository hosts the implementation of **CA-FuseNet (Crowd-Aware Fusion Network)**, an advanced multimodal deep learning architecture designed for **Composite Human Activity Recognition (CHAR)** in challenging, real-world scenarios.

CA-FuseNet leverages the complementary strengths of video stream data and human pose estimation to achieve robust and superior performance, particularly in crowd-dense environments.

## 🌟 Project Overview

The core objective of this project is to significantly advance the state-of-the-art in CHAR by developing a novel fusion network that explicitly addresses the complexities introduced by real-world, unconstrained settings, such as visual occlusion and dense social interaction.

Our model is rigorously trained and validated on the **POLIMI-ITW-S** dataset, aiming to establish a new performance benchmark that substantially outperforms current official baselines.

### Key Contributions

* **Multimodal Fusion:** Integration of **Video (RGB/Context)** and **Pose (Skeletal Dynamics)** features through a dedicated fusion mechanism.
* **Crowd-Aware Design:** Architectural components designed to maintain recognition accuracy in crowded scenes where traditional vision models often fail.
* **SOTA Performance:** A focus on producing robust, reproducible, and publishable results on the challenging **POLIMI-ITW-S** dataset.

## 🧠 Architecture: CA-FuseNet

CA-FuseNet consists of two main branches for feature extraction, followed by a Crowd-Aware Fusion module:

1.  **Video Stream Encoder:** Processes RGB frames to capture visual context and environmental cues.
2.  **Pose Stream Encoder:** Processes sequences of estimated human keypoints to capture movement kinematics and structural dynamics.
3.  **Fusion Module:** Integrates the two sets of features, employing a mechanism (e.g., attention or gating) to weigh the contribution of each modality based on the complexity and context of the scene.

## 💾 Dataset

The project relies on the **POLIMI-ITW-S** dataset (details on access and structure should be specified here if the dataset is public).

* **Dataset Name:** POLIMI-ITW-S
* **Task:** Composite Human Activity Recognition (CHAR)
* **Key Challenge:** Real-world, unconstrained videos, often involving multiple interacting subjects and significant occlusions.

## 🛠️ Setup and Installation

Follow these steps to set up the environment and reproduce our results.

### Prerequisites

* Python (3.8+)
* PyTorch (or TensorFlow, specify your framework)
* CUDA-enabled GPU (highly recommended for training)

### Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/ca-fusenet.git](https://github.com/your-username/ca-fusenet.git)
cd ca-fusenet

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt