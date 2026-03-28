from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ---------------------- YOLO Vehicle Detection ----------------------
model_yolo = YOLO("yolov8n.pt")
vehicle_classes = [
    'car','motorbike','bus','truck','auto','motorcycle','bicycle','scooter','van','pickup',
    'jeep','suv','lorry','minivan','cab','taxi','ambulance','fire engine','police car',
    'garbage truck','cement mixer','tow truck','forklift','bulldozer','excavator','road roller',
    'tractor','quad bike','snowmobile','golf cart','skateboard','segway','unicycle','rickshaw',
    'cart','tricycle','scooty','cycle rickshaw','three wheeler','four wheeler','heavy truck','light truck'
]

# ---------------------- Federated Learning Setup ----------------------
time_window = 10
lstm_epochs = 3
EDGE_NODES = 3  # simulate 3 intersections
history_files = [f"traffic_history_node{i+1}.csv" for i in range(EDGE_NODES)]

def create_lstm_model():
    model = Sequential([
        LSTM(32, input_shape=(time_window, 1), activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(0.01), loss='mse')
    return model

def detect_vehicles(frame):
    results = model_yolo.predict(frame, verbose=False)
    count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model_yolo.names[cls]
            if name in vehicle_classes:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return count, frame

# Update local history per node
def update_history(node_id, count):
    file = history_files[node_id]
    if os.path.exists(file):
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame(columns=['Vehicle_Count'])
    df = pd.concat([df, pd.DataFrame({'Vehicle_Count': [count]})], ignore_index=True)
    df.to_csv(file, index=False)

# Train local LSTM and return weights
def train_local_lstm(node_id):
    file = history_files[node_id]
    if not os.path.exists(file):
        return None
    df = pd.read_csv(file)
    if len(df) < time_window:
        return None
    data = df['Vehicle_Count'].values[-time_window:].reshape(1, time_window, 1)
    model = create_lstm_model()
    model.fit(data, data[:, -1, :], epochs=lstm_epochs, verbose=0)
    return model.get_weights()

# Federated averaging of weights
def federated_averaging(weights_list):
    if not weights_list:
        return None
    avg_weights = []
    for weights in zip(*weights_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights

# Predict using global model
global_model = create_lstm_model()
def predict_traffic():
    # Train local models
    local_weights = []
    for i in range(EDGE_NODES):
        w = train_local_lstm(i)
        if w is not None:
            local_weights.append(w)
    if local_weights:
        avg_weights = federated_averaging(local_weights)
        global_model.set_weights(avg_weights)
        # Predict based on last counts of first node (for simplicity)
        df = pd.read_csv(history_files[0])
        if len(df) >= time_window:
            data = df['Vehicle_Count'].values[-time_window:].reshape(1, time_window, 1)
            pred = int(global_model.predict(data, verbose=0)[0][0])
            return calculate_signal_time(pred)
    return 10

# ---------------------- Signal Time ----------------------
def calculate_signal_time(vehicle_count):
    if vehicle_count <= 10:
        return 10
    elif vehicle_count <= 20:
        return 20
    elif vehicle_count <= 30:
        return 30
    elif vehicle_count <= 40:
        return 40
    else:
        return 50

# ---------------------- Flask Routes ----------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    file = request.files['file']
    if not file:
        return redirect(url_for('index'))
    filename = file.filename
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"processed_{filename}")
    file.save(upload_path)

    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        frame = cv2.imread(upload_path)
        count, processed_frame = detect_vehicles(frame)
        cv2.imwrite(output_path, processed_frame)
        node_id = 0
        update_history(node_id, count)
    else:
        cap = cv2.VideoCapture(upload_path)
        total_counts = []
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0,
                              (int(cap.get(3)), int(cap.get(4))))
        node_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count, processed_frame = detect_vehicles(frame)
            total_counts.append(count)
            update_history(node_id, count)
            out.write(processed_frame)
        cap.release()
        out.release()
        count = int(np.mean(total_counts)) if total_counts else 0

    signal_time = predict_traffic()

    return render_template('result.html',
                           count=count,
                           signal_time=signal_time,
                           file_processed=f"processed_{filename}")

# ---------------------- Real-Time Webcam ----------------------
def gen_frames():
    cap = cv2.VideoCapture(0)
    last_update_time = time.time()
    current_green_time = 10
    node_id = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        count, processed_frame = detect_vehicles(frame)
        update_history(node_id, count)

        # Update predictive green time every 5 seconds
        if time.time() - last_update_time > 5:
            current_green_time = predict_traffic()
            last_update_time = time.time()

        cv2.putText(processed_frame, f"Vehicles: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Predicted Green Time: {current_green_time}s", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------------- Run Flask ----------------------
if __name__ == "__main__":
    app.run(debug=True)
