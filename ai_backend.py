from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import threading
import time
import base64 # यदि आप अलर्ट के साथ इमेज भेजना चाहते हैं

app = Flask(__name__)

# --- Global Variables for Monitoring ---
# यह डिक्शनरी प्रत्येक CCTV स्ट्रीम के लिए मॉनिटरिंग स्थिति रखेगी
# Key: rtsp_url, Value: { 'thread': Thread_obj, 'status': 'learning'/'monitoring'/'stopped',
#                          'baseline_data': {...}, 'alert_triggered': False }
monitoring_streams = {}

# --- Configuration ---
LEARNING_PERIOD_SECONDS = 300 # 5 मिनट का लर्निंग पीरियड (डेमो के लिए कम, वास्तविक में 1 घंटा या अधिक)
ALERT_THRESHOLD_PIXELS = 10  # पिक्सल परिवर्तन के लिए थ्रेशोल्ड (ट्यून करें)
ALERT_COOLDOWN_SECONDS = 60 # एक बार अलर्ट होने के बाद अगले अलर्ट के लिए वेट टाइम

# --- Anomaly Detection Logic ---
def analyze_stream_for_anomalies(rtsp_url, user_id):
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"Error: Could not open video stream for {rtsp_url}")
        monitoring_streams[rtsp_url]['status'] = 'stopped'
        return

    # Initialize monitoring state for this stream
    monitoring_streams[rtsp_url] = {
        'status': 'learning',
        'baseline_data': {
            'avg_brightness_history': [],
            'avg_motion_history': [],
            # आप और मेट्रिक्स यहाँ जोड़ सकते हैं
        },
        'alert_triggered': False,
        'last_alert_time': 0,
        'user_id': user_id,
        'thread': threading.current_thread() # अपने थ्रेड को संदर्भित करें
    }
    stream_state = monitoring_streams[rtsp_url]
    
    # Placeholder for baseline data (e.g., last 'n' frames data)
    baseline_frames_data = [] 
    max_baseline_frames = 100 # सीखने के लिए कितने फ्रेम का औसत लेना है

    print(f"[{rtsp_url}] Starting LEARNING phase for {LEARNING_PERIOD_SECONDS} seconds...")
    learning_start_time = time.time()
    last_frame = None

    while stream_state['status'] == 'learning' and (time.time() - learning_start_time < LEARNING_PERIOD_SECONDS):
        ret, frame = cap.read()
        if not ret:
            print(f"[{rtsp_url}] Learning phase: Stream ended or error.")
            stream_state['status'] = 'stopped'
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_blur = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        
        # Calculate motion for learning baseline
        if last_frame is not None:
            frame_diff = cv2.absdiff(last_frame, gray_frame_blur)
            motion_score = np.mean(frame_diff)
            stream_state['baseline_data']['avg_motion_history'].append(motion_score)
            if len(stream_state['baseline_data']['avg_motion_history']) > max_baseline_frames:
                stream_state['baseline_data']['avg_motion_history'].pop(0)

        stream_state['baseline_data']['avg_brightness_history'].append(np.mean(gray_frame))
        if len(stream_state['baseline_data']['avg_brightness_history']) > max_baseline_frames:
            stream_state['baseline_data']['avg_brightness_history'].pop(0)
            
        last_frame = gray_frame_blur
        time.sleep(0.1) # फ्रेम प्रोसेसिंग के बीच थोड़ा विराम

    if stream_state['status'] != 'stopped':
        # Calculate actual baseline averages after learning
        if stream_state['baseline_data']['avg_motion_history']:
            stream_state['baseline_motion'] = np.mean(stream_state['baseline_data']['avg_motion_history'])
            stream_state['baseline_motion_std'] = np.std(stream_state['baseline_data']['avg_motion_history'])
        else:
            stream_state['baseline_motion'] = 0
            stream_state['baseline_motion_std'] = 1 # avoid division by zero
            
        if stream_state['baseline_data']['avg_brightness_history']:
            stream_state['baseline_brightness'] = np.mean(stream_state['baseline_data']['avg_brightness_history'])
            stream_state['baseline_brightness_std'] = np.std(stream_state['baseline_data']['avg_brightness_history'])
        else:
            stream_state['baseline_brightness'] = 127 # Mid-gray
            stream_state['baseline_brightness_std'] = 1 

        stream_state['status'] = 'monitoring'
        print(f"[{rtsp_url}] LEARNING phase complete. Switching to MONITORING. Baseline Motion: {stream_state['baseline_motion']:.2f}, Brightness: {stream_state['baseline_brightness']:.2f}")

    # --- MONITORING phase ---
    last_frame = None # Reset last_frame for monitoring
    while stream_state['status'] == 'monitoring':
        ret, frame = cap.read()
        if not ret:
            print(f"[{rtsp_url}] Monitoring phase: Stream ended or error. Attempting reconnect in 5s...")
            # Implement re-connection logic or immediate alert for stream loss
            trigger_alert(rtsp_url, "STREAM_INTERRUPTED", "Video stream interrupted or ended unexpectedly.", frame)
            stream_state['status'] = 'stopped' # For simplicity, stop if stream breaks
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_blur = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        
        is_anomaly = False
        anomaly_type = "UNKNOWN"
        anomaly_details = ""

        # 1. Motion Anomaly Detection
        if last_frame is not None:
            frame_diff = cv2.absdiff(last_frame, gray_frame_blur)
            current_motion_score = np.mean(frame_diff)
            
            # Simple Z-score like check for motion (adjust threshold as needed)
            # if abs(current_motion_score - stream_state['baseline_motion']) > stream_state['baseline_motion_std'] * 3: # 3 standard deviations
            # This is too sensitive for sudden large changes. Let's use absolute diff for now.
            
            # If current motion is drastically different from baseline
            # This covers both very high motion (camera moved, object blocking)
            # and very low motion (stream frozen, camera completely covered and no light changes)
            if abs(current_motion_score - stream_state['baseline_motion']) > ALERT_THRESHOLD_PIXELS:
                is_anomaly = True
                anomaly_type = "ABNORMAL_MOTION"
                anomaly_details = f"Motion score changed from {stream_state['baseline_motion']:.2f} to {current_motion_score:.2f}"
            
            # Additional check for complete frame freeze (zero motion for a prolonged period if baseline had motion)
            if stream_state['baseline_motion'] > 1 and current_motion_score < 0.5 and not is_anomaly: # If baseline had motion but now it's almost zero
                 is_anomaly = True
                 anomaly_type = "STREAM_FROZEN"
                 anomaly_details = "Video stream appears to be frozen."


        # 2. Brightness Anomaly Detection
        current_brightness = np.mean(gray_frame)
        if abs(current_brightness - stream_state['baseline_brightness']) > stream_state['baseline_brightness_std'] * 4: # 4 standard deviations for brightness
            is_anomaly = True
            anomaly_type = "BRIGHTNESS_CHANGE"
            anomaly_details = f"Brightness changed from {stream_state['baseline_brightness']:.2f} to {current_brightness:.2f}"

        # 3. Complete Obscuration Detection (e.g., all black or all white frame)
        if np.all(gray_frame < 5) or np.all(gray_frame > 250): # Almost completely black or white
            is_anomaly = True
            anomaly_type = "COMPLETE_OBSCURATION"
            anomaly_details = "Video stream is completely black or white."

        # Trigger alert if anomaly detected
        if is_anomaly:
            if not stream_state['alert_triggered'] or (time.time() - stream_state['last_alert_time'] > ALERT_COOLDOWN_SECONDS):
                print(f"[{rtsp_url}] ANOMALY DETECTED: {anomaly_type} - {anomaly_details}")
                trigger_alert(rtsp_url, anomaly_type, anomaly_details, frame)
                stream_state['alert_triggered'] = True
                stream_state['last_alert_time'] = time.time()
        else:
            stream_state['alert_triggered'] = False # Reset alert flag if no anomaly

        last_frame = gray_frame_blur
        time.sleep(0.1) # फ्रेम प्रोसेसिंग के बीच थोड़ा विराम

    print(f"[{rtsp_url}] Monitoring stopped.")
    cap.release()
    if rtsp_url in monitoring_streams:
        del monitoring_streams[rtsp_url] # Remove from active monitoring

def trigger_alert(rtsp_url, anomaly_type, details, frame):
    # --- This is where you'd send an alert to your Flutter app ---
    # For now, we'll just print it and simulate an API call or webhook.
    
    # Encode frame to base64 if you want to send image with alert
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    alert_data = {
        'rtsp_url': rtsp_url,
        'user_id': monitoring_streams[rtsp_url]['user_id'],
        'anomaly_type': anomaly_type,
        'details': details,
        'timestamp': time.time(),
        'image': encoded_image # Optional: send image of the anomaly
    }
    
    print(f"--- ALERT for {rtsp_url} ({monitoring_streams[rtsp_url]['user_id']}): {anomaly_type} ---")
    print(f"Details: {details}")
    # Here you would make an HTTP POST request to your Flutter app's notification endpoint
    # Or send it to a message queue like RabbitMQ, or a Firebase Cloud Message.
    # For example:
    # import requests
    # requests.post("YOUR_FLUTTER_APP_NOTIFICATION_ENDPOINT", json=alert_data)


# --- Flask API Endpoints ---
@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    data = request.json
    rtsp_url = data.get('rtsp_url')
    user_id = data.get('user_id', 'unknown_user')

    if not rtsp_url:
        return jsonify({"status": "error", "message": "RTSP URL is required"}), 400

    if rtsp_url in monitoring_streams and monitoring_streams[rtsp_url]['status'] != 'stopped':
        return jsonify({"status": "warning", "message": f"Monitoring already active for {rtsp_url}"}), 200

    # Start the monitoring in a new thread
    thread = threading.Thread(target=analyze_stream_for_anomalies, args=(rtsp_url, user_id))
    thread.daemon = True # Allow main program to exit even if thread is running
    thread.start()
    
    # Store thread object and initial state
    monitoring_streams[rtsp_url] = {
        'thread': thread,
        'status': 'starting',
        'alert_triggered': False,
        'user_id': user_id,
        'last_alert_time': 0,
        'baseline_data': {} # Will be populated by the thread
    }

    return jsonify({"status": "success", "message": f"Started monitoring for {rtsp_url}. Learning phase initiated."})

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    data = request.json
    rtsp_url = data.get('rtsp_url')

    if not rtsp_url:
        return jsonify({"status": "error", "message": "RTSP URL is required"}), 400

    if rtsp_url in monitoring_streams:
        monitoring_streams[rtsp_url]['status'] = 'stopped' # Signal the thread to stop
        # thread.join() # Don't join here, it will block the API call
        return jsonify({"status": "success", "message": f"Monitoring stopped for {rtsp_url}"})
    else:
        return jsonify({"status": "warning", "message": f"No active monitoring found for {rtsp_url}"}), 200

@app.route('/monitoring_status', methods=['GET'])
def get_monitoring_status():
    status_list = []
    for url, data in monitoring_streams.items():
        status_list.append({
            'rtsp_url': url,
            'status': data['status'],
            'alert_triggered': data['alert_triggered'],
            'user_id': data['user_id'],
            'last_alert_time': data['last_alert_time']
        })
    return jsonify(status_list)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/services')
def services():
    # यह 'templates/services.html' (या 'index.html' के अंदर services section) को लोड करेगा
    return render_template('service.html')

# 3. Download App Page Route
@app.route('/download')
def download():
    # यह 'templates/download.html' को लोड करेगा
    return render_template('downloadapp.html')

# 4. About Us Page Route (अगर इसे अलग URL पर रखा जाए)
@app.route('/about')
def about():
    # यह 'templates/about.html' को लोड करेगा
    return render_template('about us.html')    

if __name__ == '__main__':
    # Flask app runs on localhost:5000 by default
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False) 
    # use_reloader=False because threading can cause issues with reloader