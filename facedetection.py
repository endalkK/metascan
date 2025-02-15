import cv2
import os
import sqlite3
import time

# Load OpenCV's pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a directory to store captured faces
output_folder = 'captured_faces'
os.makedirs(output_folder, exist_ok=True)

# Connect to SQLite database (or create if not exists)
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()

# Create table to store face images if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        timestamp TEXT
    )
''')
conn.commit()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set to track detected faces to avoid duplicates
detected_faces = set()
reset_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_id = (x, y, w, h)  # Unique identifier for a detected face

        if face_id not in detected_faces:
            detected_faces.add(face_id)

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract face region
            face_img = frame[y:y+h, x:x+w]

            # Save face image
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{output_folder}/face_{timestamp}.jpg"
            cv2.imwrite(filename, face_img)

            # Store in database
            cursor.execute("INSERT INTO faces (filename, timestamp) VALUES (?, ?)", (filename, timestamp))
            conn.commit()
            print(f"Saved: {filename}")

    # Show video stream with detection
    cv2.imshow('Live Face Detection', frame)

    # Reset detected faces every 10 seconds (to allow new captures if person moves)
    if time.time() - reset_time > 10:
        detected_faces.clear()
        reset_time = time.time()

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
