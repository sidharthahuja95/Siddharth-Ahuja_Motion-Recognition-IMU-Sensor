
# Motion Recognition using IMU Sensor Fusion on Raspberry Pi

This project implements a real-time gesture classification system using **Raspberry Pi 4** and **Sense HAT**. It uses **IMU (Inertial Measurement Unit)** data—captured via accelerometer and gyroscope—to recognize four motion types: `move_none`, `move_circle`, `move_shake`, and `move_twist`. The model is trained using TensorFlow/Keras and deployed with **TensorFlow Lite** for efficient edge inference.

---

## Objectives

- Collect real-time IMU sensor data from Raspberry Pi
- Preprocess and normalize time-series data
- Train a deep learning model for gesture classification
- Convert the model to `.tflite` format for deployment
- Provide real-time motion feedback using the Sense HAT LED matrix

---

## Key Features

- 2-second motion windows at 50Hz sampling rate (100 time steps × 6 features)
- Neural network architecture with dropout and batch normalization
- Real-time inference on Raspberry Pi using TensorFlow Lite
- Visual feedback using LED matrix (color-coded gestures)
- Custom dataset of 250 labeled motion samples

---

## Repository Structure

```bash
Siddharth-Ahuja_Motion-Recognition-using-IMU-Sensor-Fusion-on-Raspberry-Pi/
├── data_collection/
│   └── collect_imu_data.py           # Script for collecting IMU data
├── motion_data/
│   ├── move_none/
│   ├── move_circle/
│   ├── move_shake/
│   └── move_twist/
├── model_training/
│   └── imu_motion_classifier.ipynb   # Google Colab notebook
├── model/
│   ├── motion_model.tflite           # TFLite model for deployment
│   └── best_model.h5                 # Checkpoint model
├── demo_pictures/
│   └── *.png                         # Screenshots of results and motion samples
├── README.md
└── requirements.txt                  # Python dependencies
```

---

## Dataset Overview

- Total Samples: **250**
- Classes:
  - `move_none`: No motion
  - `move_circle`: Hand moves in a circle
  - `move_shake`: Quick back-and-forth motion
  - `move_twist`: Wrist twist motion
- Features: 3-axis accelerometer + 3-axis gyroscope = 6 features
- Sample Length: 2 seconds @ 50 Hz = 100 time steps (flattened to 300 features)

---

## Technologies Used

- **Python 3.8+**
- **TensorFlow / Keras**
- **Google Colab** (Training + Preprocessing)
- **NumPy, Scikit-learn, Matplotlib**
- **Raspberry Pi 4 (4 GB RAM)**
- **Sense HAT with 9-DOF IMU**

---

## Deployment Workflow

1. **Data Collection**  
   Run `collect_imu_data.py` on Raspberry Pi to capture labeled `.npy` IMU samples.

2. **Model Training**  
   Use `imu_motion_classifier.ipynb` on Colab:
   - Preprocess and normalize IMU samples
   - Train the model (128 → 64 → softmax)
   - Save the best model (`.h5`)
   - Convert to `.tflite` for Raspberry Pi

3. **Real-time Inference**  
   - Load `motion_model.tflite` on Raspberry Pi
   - Use IMU data stream to predict motion
   - Display corresponding color on LED:
     - blue colour `move_twist`
     - green colour `move_shake`
     - red colour `move_circle`
     - no colour `move_none`

---

##  Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/sidharthahuja95/Siddharth-Ahuja_Motion-Recognition-using-IMU-Sensor-Fusion-on-Raspberry-Pi
   cd Siddharth-Ahuja_Motion-Recognition-using-IMU-Sensor-Fusion-on-Raspberry-Pi
   ```

2. (Optional) Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Collect new IMU data using:
   ```bash
   python data_collection/collect_imu_data.py
   ```

4. Upload `.npy` files to Google Colab and run `imu_motion_classifier.ipynb` to retrain or fine-tune the model.

5. Copy `motion_model.tflite` back to Pi and run the inference script for real-time gesture classification.

---

##  Results Snapshot

- Achieved validation accuracy of **85–90%**
- Real-time classification with <1 sec response delay
- IMU plots and confusion matrix included in report

---

## Visual Demo (Images)

| Gesture        | Live Detection Screenshot        |
|----------------|----------------------------------|
| `move_none`    | ![](demo_pictures/mn1.png)       |
| `move_circle`  | ![](demo_pictures/mc1.png)       |
| `move_shake`   | ![](demo_pictures/ms1.png)       |
| `move_twist`   | ![](demo_pictures/mt1.png)       |

---

## Future Improvements

- Train on larger, multi-user datasets for better generalization
- Add more gesture classes (e.g., swipe, tap, double-tap)
- Implement quantization or Coral TPU for faster inference
- Integrate GUI for gesture visualization

---

##  References

- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [Sense HAT API Docs](https://pythonhosted.org/sense-hat/)
- [Google Colab](https://colab.research.google.com/)
- Ian Goodfellow, *Deep Learning*, MIT Press, 2016

---

## Acknowledgements

Project developed for **Embedded Systems Lab** at Deggendorf Institute of Technology (Cham Campus), supervised by **Prof. Dr. Tobias Schäffer**.
