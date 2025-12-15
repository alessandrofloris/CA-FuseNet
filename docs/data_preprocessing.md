## ⚙️ Data Preprocessing Documentation

This section details the structure and contents of the preprocessed data files used for model training and evaluation. These files are generated using a customized fork of the **PyHAPT** project, which processes raw JSON annotations from the **ITW-POLIMI** dataset.

### 1. Source and Generation

* **Source Data:** Raw JSON annotation files from the **ITW-POLIMI** dataset.
* **Preprocessing Tool:** A custom fork of the [PyHAPT project](https://github.com/alessandrofloris/PyHAPT).
* **Original PyHAPT Functionality:** Originally designed to output the train/test split containing skeleton information and action labels.
* **Current Extension:** The project has been extended to incorporate additional metadata necessary for creating RGB tublets and crowd data.

### 2. Data Tensor Structures

The preprocessed data is stored in two main NumPy arrays (`train_data_joint` and `train_data_bbox`) and one metadata file (`train_label`). All numerical data within the arrays are stored as **`float64`**.

#### A. `train_data_joint` (Skeleton Keypoints)

This tensor contains the raw normalized keypoint coordinates for the human poses in the sequences.

| Dimension | Abbreviation | Description | Example Value |
| :---: | :---: | :--- | :--- |
| **0** | N | Number of samples (individual action sequences). | 27 |
| **1** | C | Coordinate dimensions (X and Y coordinates of the keypoints). | 2 |
| **2** | T | Number of temporal frames in the action sequence (padded). | 300 |
| **3** | V | Number of keypoint pairs (e.g., body joints). | 17 |
| **4** | M | Person ID dimension (contains the ID of the person performing the action). | 1 |

* **Example Shape (with 27 sequences):** $(27, 2, 300, 17, 1)$

#### B. `train_data_bbox` (Bounding Boxes)

This tensor contains the coordinates of the bounding box surrounding the person for each frame.

| Dimension | Abbreviation | Description | Example Value |
| :---: | :---: | :--- | :--- |
| **0** | N | Number of samples (individual action sequences). | 27 |
| **1** | T | Number of temporal frames in the action sequence (padded). | 300 |
| **2** | B | Bounding Box coordinates, formatted as $(x_1, y_1, x_2, y_2)$. | 4 |

* **Example Shape (with 27 sequences):** $(27, 300, 4)$

### 3. `train_label` (Metadata)

This is a **pickle file** containing metadata associated with each action sequence sample.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| **`sample_name`** | String | The concatenation of the original video name and the person ID performing the action (e.g., `video_123_person_0`). |
| **`label`** | Integer | The integer encoding for the specific action being performed (e.g., `1` for "walking", `2` for "waving"). |
| **`frame_indices`** | List[Int] | A list of frame indices (padded to 300) that define the specific action segment within the video. |
| **`video_paths`** | String | The relative path to the source video (e.g., `action_class/video_name.mp4`). |