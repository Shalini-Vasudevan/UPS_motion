
# Multi-Person 2D/3D Pose Estimation and Tracking

This project provides a comprehensive pipeline for detecting, tracking, and analyzing the pose of multiple people in a video. It integrates state-of-the-art models for 2D and 3D pose estimation and provides a modular framework for easy extension.


---
## ## Features
* **Multi-Person Detection & Tracking:** Utilizes the YOLO object detector and a robust tracking algorithm to identify and assign persistent IDs to every person in a video.
* **High-Performance 2D Pose Estimation:** Integrates multiple 2D pose models, including the efficient **RTM-Pose** and the accurate **ViT-Pose**.
* **3D Pose Lifting:** Lifts the extracted 2D keypoints into 3D space for advanced motion analysis.
* **Modular Architecture:** The code is organized into `core_modules` for easy maintenance and swapping of different models.

---
## ## Technology Stack
* **Detection:** YOLO (You Only Look Once)
* **Tracking:** ByteTrack (or SORT, depending on `detection_tracking.py` implementation)
* **2D Pose Models:** RTM-Pose, ViT-Pose
* **3D Pose Lifting:** MotionAGFormer, LITE (or similar)
* **Core Libraries:** PyTorch, OpenCV, Ultralytics, MMPose

---
## ## Project Structure
The repository is organized into a modular structure for clarity and scalability.

```

├── core\_modules/
│   ├── detection\_tracking.py     \# Handles person detection and tracking
│   ├── pose\_2d\_rtmpose.py        \# 2D pose estimation with RTM-Pose
│   ├── pose\_2d\_vitpose.py        \# 2D pose estimation with ViT-Pose
│   ├── pose\_3d.py                \# 2D-to-3D pose lifting
│   └── video\_utils.py            \# Utility functions for video I/O
├── demo/                         \# Folder for demo videos and outputs
├── main\_driver.py                \# Main script to run the full pipeline
└── requirements.txt              \# Project dependencies

````

---
## ## Installation
1.  **Clone the repository:**
    ```bash
    git clone <your-repository-link>
    cd <your-repository-name>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have the necessary build tools and CUDA installed if you are using a GPU).*

---
## ## Usage
The main entry point for the project is `main_driver.py`. You can run the full pipeline from your terminal.

1.  Place your input video inside the `demo/video/` directory.

2.  Run the main driver script, specifying the input video and desired output path.
    ```bash
    python main_driver.py --input_video demo/video/your_video.mp4 --output_video demo/output/result.mp4
    ```

3.  The script will process the video and save the final output with pose tracking overlays to the specified path.
````
