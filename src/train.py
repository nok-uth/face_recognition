import cv2
import numpy as np
import os
import json

def train_model():
    data_path = '../data/known_faces'
    
    # The AI only understands numbers, so we create a dictionary to map names to ID numbers
    name_to_id = {}
    current_id = 0
    
    face_samples = []
    ids = []
    
    # Load the face detector so we can crop just the face out of your screenshots
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Scanning image folders...")
    
    # Loop through every folder and file in known_faces
    for root, dirs, files in os.walk(data_path):
        for file in files:
            # We specifically tell it to look for png, jpg, and jpeg files
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                path = os.path.join(root, file)
                person_name = os.path.basename(root)
                
                # If we haven't seen this person yet, assign them a new ID number
                if person_name not in name_to_id:
                    name_to_id[person_name] = current_id
                    current_id += 1
                    
                id_num = name_to_id[person_name]
                
                # Load the screenshot and convert to grayscale
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect the face in the screenshot
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                
                for (x, y, w, h) in faces:
                    # Crop out just the face pixels and add it to our training data
                    face_samples.append(gray[y:y+h, x:x+w])
                    ids.append(id_num)

    if len(face_samples) == 0:
        print("No faces found in your screenshots! Make sure they are clear.")
        return

    print(f"Training on {len(face_samples)} faces. This might take a few seconds...")
    
    # Initialize the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Train the algorithm
    recognizer.train(face_samples, np.array(ids))
    
    # Save the trained model and the name dictionary into the data folder
    recognizer.write('../data/face_trainer.yml')
    with open('../data/names.json', 'w') as f:
        json.dump(name_to_id, f)
        
    print("Training complete! Brain saved to data folder.")

if __name__ == "__main__":
    train_model()