import cv2
import glob
import os
import json
import math
from collections import Counter
from database import log_attendance

def process_videos():
    video_files = glob.glob("../data/input_videos/*.mp4") + glob.glob("../data/input_videos/*.mov")
    if len(video_files) == 0:
        print("No videos found!")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    try:
        recognizer.read('../data/face_trainer.yml')
    except:
        print("Could not find the brain file! Did you run train.py?")
        return

    with open('../data/names.json', 'r') as f:
        name_to_id = json.load(f)
        id_to_name = {id_num: name for name, id_num in name_to_id.items()}

    for video_path in video_files:
        file_name = os.path.basename(video_path)
        print(f"--- Scanning Video: {file_name} ---")
        
        cap = cv2.VideoCapture(video_path)
        
        # --- NEW: Memory for Temporal Smoothing ---
        # This list will hold data for faces currently on screen.
        active_faces = [] 
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            new_active_faces = [] # Reset trackers for the current frame
            
            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y+h, x:x+w]
                id_num, distance = recognizer.predict(face_roi)
                
                # 1. Get the raw, unfiltered guess for this exact millisecond
                if distance < 130:
                    raw_guess = id_to_name[id_num]
                else:
                    raw_guess = "Unknown"
                    
                # 2. Find the mathematical center point of the face
                cx, cy = x + (w // 2), y + (h // 2)
                
                # 3. Check if we saw this face in the exact same spot in the last frame
                matched_tracker = None
                for tracker in active_faces:
                    # If a face is within 50 pixels of where it was last frame, it's the same person
                    dist = math.hypot(cx - tracker['center'][0], cy - tracker['center'][1])
                    if dist < 50:
                        matched_tracker = tracker
                        break
                        
                # 4. Update the 5-frame memory history for this face
                if matched_tracker:
                    history = matched_tracker['names']
                    history.append(raw_guess)
                    if len(history) > 5:
                        history.pop(0) # Kick out the oldest frame so we only keep the last 5
                else:
                    history = [raw_guess] # First time seeing this face
                    
                new_active_faces.append({'center': (cx, cy), 'names': history})
                
                # 5. THE MAGIC: Take a majority vote from the last 5 frames!
                most_common_guess = Counter(history).most_common(1)[0][0]
                
                # Set colors and log attendance based on the SMOOTHED guess
                if most_common_guess == "Unknown":
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                    log_attendance(most_common_guess) 
                
                # Draw the smooth, flicker-free box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, most_common_guess, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Pass the memories onto the next frame
            active_faces = new_active_faces

            # Show the video feed on screen
            cv2.imshow('Attendance Scanner', frame)
            
            # Press 'q' to quit (Wait 30ms between frames to simulate normal video speed)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
        cap.release()
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_videos()