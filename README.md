# Driver Fatigue Detection System

## Overview

This project implements a real-time driver fatigue detection system using computer vision. It analyzes a live webcam feed to monitor key indicators of drowsiness—eye blinks (including slow blinks and microsleeps) and yawning—and provides a real-time "Fatigue Level" classification to alert users.

## Key Features

* **Real-time Analysis:** Continuous monitoring of facial landmarks for drowsiness.
* **Comprehensive Detection:** Utilizes Eye Aspect Ratio (EAR) for blink detection (normal, slow, microsleeps) and Mouth Aspect Ratio (MAR) for yawning.
* **Enhanced Accuracy:** Incorporates head pose estimation, adaptive low-light adjustment, and glasses detection to improve reliability.
* **Fatigue Level Classification:** Outputs a clear fatigue status (Normal, Mild Drowsy, Drowsy, Very Drowsy).

## Technologies Used

* **Python 3.x**
* **OpenCV (`cv2`)**: Video processing.
* **MediaPipe (`mediapipe`)**: Facial landmark detection.
* **NumPy (`numpy`)**: Numerical operations.

## Setup & Run

### Prerequisites

* Python 3.x installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/drivers-fatigue-detection.git](https://github.com/YOUR_USERNAME/drivers-fatigue-detection.git)
    cd drivers-fatigue-detection
    ```
    *(Replace `YOUR_USERNAME` and `drivers-fatigue-detection` with your actual details.)*
2.  **Install dependencies:**
    ```bash
    pip install opencv-python mediapipe numpy
    ```

### How to Run

1.  **Navigate to the project directory:**
    ```bash
    cd /path/to/your/project/folder
    ```
2.  **Execute the script:**
    ```bash
    python main.py
    ```
    Press `q` to quit the application.

## Algorithm Insights

The system calculates **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** from MediaPipe's 3D facial landmarks. These ratios, combined with blink duration, yawn duration, and head pose data, feed into a classification logic to determine fatigue. Dynamic thresholds adjust for head movements and glasses, while adaptive brightness handles varying lighting conditions.

## License

[YOUR TEXT HERE: e.g., "This project is licensed under the MIT License." or "MIT License - see the LICENSE.md file for details."]

## Contact

[YOUR TEXT HERE: e.g., Your Name - your.email@example.com - LinkedIn Profile (Optional)]

Project Link: [https://github.com/YOUR_USERNAME/drivers-fatigue-detection](https://github.com/YOUR_USERNAME/drivers-fatigue-detection)
