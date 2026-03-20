# Real-Time Facial Recognition & Attendance System

A lightweight, real-time computer vision application that detects, tracks, and recognizes faces from video feeds, automatically logging identified individuals into an SQLite database. 

This project completely bypasses heavy cloud APIs in favor of local, edge-computed machine learning models, specifically utilizing Haar Cascades for rapid object detection and Local Binary Patterns Histograms (LBPH) for facial feature extraction and identification.

##  Technical Architecture

* **Computer Vision Engine:** OpenCV (`opencv-contrib-python`)
* **Detection Model:** Haar Cascade (Frontal Face Default)
* **Recognition Algorithm:** LBPH (Local Binary Patterns Histograms)
* **Database:** SQLite3
* **Language:** Python 3.x

##  Key Features

* **Temporal Smoothing (Debouncing):** Implemented state management and centroid tracking to maintain a 5-frame rolling memory of detected faces. The system uses a "majority vote" algorithm to completely eliminate single-frame prediction flickering caused by environmental noise or blinks.
* **Optimized Processing:** Tuned the scale factors, `minNeighbors`, and spatial thresholds to drastically reduce false positives (ghost boxes) while maintaining high frame rates on standard CPU hardware.
* **Idempotent Database Logging:** Engineered the SQLite integration to reject duplicate entries for the same user on the same date, protecting the database from high-frequency read/write spam during continuous video processing.

##  How to Run Locally

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

# 2. Set up the virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

# 3. Train the Model 
# (Note: Add clear, front-facing images to data/known_faces/NAME/ first!)
cd src
python train.py

# 4. Run the Scanner 
# (Note: Drop an .mp4 or .mov file into data/input_videos/ first!)
python vision.py
