import tensorflow as tf
import numpy as np
import time
from sense_hat import SenseHat

sense = SenseHat()
sense.clear()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="motion_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

LABELS = ["move_none", "move_circle", "move_shake", "move_twist"]
COLORS = {
    "move_none": [0, 0, 0],         # Black/off
    "move_circle": [255, 0, 0],     # Red
    "move_shake": [0, 255, 0],      # Green
    "move_twist": [0, 0, 255]       # Blue
}

SAMPLES = 50
FREQ_HZ = 50
DELAY = 1.0 / FREQ_HZ

# Thresholds
BASE_CONFIDENCE_THRESHOLD = 0.80     # General minimum threshold
CIRCLE_THRESHOLD = 0.92              # Higher threshold for move_circle
NOISE_THRESHOLD = 0.06               # Dead zone for IMU noise

def dead_zone(value, threshold=NOISE_THRESHOLD):
    return 0 if abs(value) < threshold else value

def read_imu_sample():
    acc = sense.get_accelerometer_raw()
    gyro = sense.get_gyroscope_raw()
    time.sleep(DELAY)

    # Apply dead zone to reduce noise
    acc_x = dead_zone(acc['x'])
    acc_y = dead_zone(acc['y'])
    acc_z = dead_zone(acc['z'])
    gyro_x = dead_zone(gyro['x'])
    gyro_y = dead_zone(gyro['y'])
    gyro_z = dead_zone(gyro['z'])

    return [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]

try:
    while True:
        print("Collecting 1s sample ...")
        samples = [read_imu_sample() for _ in range(SAMPLES)]
        input_data = np.array(samples).flatten().astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]
        print("Raw model output:", output)

        predicted_index = int(np.argmax(output))
        confidence = output[predicted_index]

        # Dynamic threshold logic
        if predicted_index == 1 and confidence >= CIRCLE_THRESHOLD:
            label = LABELS[predicted_index]
        elif confidence >= BASE_CONFIDENCE_THRESHOLD:
            label = LABELS[predicted_index]
        else:
            label = "move_none"

        print(f"Predicted: {label} with confidence {confidence:.3f}")
        sense.clear(COLORS[label])

except KeyboardInterrupt:
    sense.clear()
    print("Stopped.")
