import cv2
import numpy as np

def calculate_ratio(a, b):
    """Calculate the ratio of two lengths."""
    return max(a, b) / min(a, b)

def calculate_golden_ratio_score(ratios):
    """Calculate an overall score based on how close the ratios are to the golden ratio (1.618)."""
    golden_ratio = 1.618
    score = 0
    for ratio in ratios.values():
        score += 1 - abs(golden_ratio - ratio) / golden_ratio
    return round((score / len(ratios)) * 10, 2)  # Normalize to a score out of 10

def analyze_face_from_camera():
    """Analyze the golden ratio of a face using live camera input."""
    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera. Please check if the camera is connected and not in use by another application.")
        return

    # Load OpenCV's pre-trained Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Please ensure the camera is functioning correctly.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Define placeholder points for analysis
            points = np.array([
                [x + int(w * 0.3), y + int(h * 0.4)],  # Approx eye corner left
                [x + int(w * 0.7), y + int(h * 0.4)],  # Approx eye corner right
                [x + int(w * 0.5), y + int(h * 0.7)],  # Bottom of nose
                [x + int(w * 0.5), y + int(h * 0.2)],  # Top of nose bridge
                [x + int(w * 0.2), y + int(h * 0.85)], # Mouth corner left
                [x + int(w * 0.8), y + int(h * 0.85)], # Mouth corner right
                [x + int(w * 0.5), y + h]              # Bottom of chin
            ])

            # Calculate key measurements for golden ratio analysis
            eye_width = np.linalg.norm(points[1] - points[0])  # Distance between outer corners of eyes
            nose_length = np.linalg.norm(points[3] - points[2])  # Distance from forehead to bottom of nose
            mouth_width = np.linalg.norm(points[5] - points[4])  # Distance between mouth corners
            face_height = np.linalg.norm(points[6] - points[3])  # Distance from chin to forehead

            # Calculate ratios
            ratios = {
                "Eye to Nose": calculate_ratio(eye_width, nose_length),
                "Mouth to Nose": calculate_ratio(mouth_width, nose_length),
                "Face Height to Width": calculate_ratio(face_height, eye_width),
            }

            # Calculate overall score
            score = calculate_golden_ratio_score(ratios)

            # Draw grid lines on the face
            cv2.line(frame, (x + int(w * 0.33), y), (x + int(w * 0.33), y + h), (255, 255, 0), 1)
            cv2.line(frame, (x + int(w * 0.66), y), (x + int(w * 0.66), y + h), (255, 255, 0), 1)
            cv2.line(frame, (x, y + int(h * 0.33)), (x + w, y + int(h * 0.33)), (255, 255, 0), 1)
            cv2.line(frame, (x, y + int(h * 0.66)), (x + w, y + int(h * 0.66)), (255, 255, 0), 1)

            # Display ratios and score on the frame
            y_offset = 20
            for feature, ratio in ratios.items():
                text = f"{feature}: {ratio:.3f}"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            score_text = f"Score: {score}"
            cv2.putText(frame, score_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw facial landmarks
            for (px, py) in points:
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow("Face Analysis", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
analyze_face_from_camera()
