import cv2
import mediapipe as mp
import torch
import numpy as np
from train_model import SignLanguageAttentionModel
from dataconvertor import preprocess_image
import time

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode, 
            max_num_hands=self.maxHands, 
            min_detection_confidence=self.detectionCon, 
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True, color=(255, 0, 255), z_axis=False):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape

            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cz = round(lm.z, 3) if z_axis else None
                lmList.append([id, cx, cy] if not z_axis else [id, cx, cy, cz])

                if draw:
                    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

        return lmList

    def getLandmarks(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[0]
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            return landmarks
        return None

def enhance_contrast(image):
    """Apply advanced contrast enhancement"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

def detect_skin(image):
    """Detect skin color in various color spaces"""
    # Convert to different color spaces
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Define skin color ranges
    rgb_mask = ((rgb_img[:,:,0] > 95) & (rgb_img[:,:,1] > 40) & (rgb_img[:,:,2] > 20) &
                (rgb_img[:,:,0] - rgb_img[:,:,1] > 15) & (rgb_img[:,:,0] - rgb_img[:,:,2] > 15))
    
    hsv_mask = ((hsv_img[:,:,0] > 0) & (hsv_img[:,:,0] < 50) &
                (hsv_img[:,:,1] > 50) & (hsv_img[:,:,1] < 170) &
                (hsv_img[:,:,2] > 50))
    
    ycrcb_mask = ((ycrcb_img[:,:,1] > 135) & (ycrcb_img[:,:,1] < 180) &
                  (ycrcb_img[:,:,2] > 85) & (ycrcb_img[:,:,2] < 135))
    
    # Combine masks
    skin_mask = (rgb_mask & hsv_mask & ycrcb_mask).astype(np.uint8) * 255
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    
    # Apply mask to image
    skin_img = cv2.bitwise_and(image, image, mask=skin_mask)
    return skin_img

def load_model(model_path='sign_language_model.pth'):
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageAttentionModel(num_classes=24).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def predict_gesture(model, image, device, detector):
    """Predict the gesture from an image"""
    # Try different preprocessing variations
    variations = []
    
    # Original image resized
    variations.append(cv2.resize(image, (640, 640)))
    
    # Enhanced contrast
    enhanced = enhance_contrast(image)
    variations.append(cv2.resize(enhanced, (640, 640)))
    
    # Skin detection
    skin_detected = detect_skin(image)
    variations.append(cv2.resize(skin_detected, (640, 640)))
    
    # Grayscale with CLAHE
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    variations.append(cv2.resize(cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR), (640, 640)))
    
    # Original preprocessing
    variations.append(preprocess_image(image))
    
    for processed_image in variations:
        landmarks = detector.getLandmarks(processed_image)
        if landmarks:
            flat_landmarks = [coord for point in landmarks for coord in point]
            
            if len(flat_landmarks) == 63:
                # Convert to tensor and add batch dimension
                landmarks_tensor = torch.tensor(flat_landmarks, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(landmarks_tensor)
                    _, predicted = torch.max(outputs, 1)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence = probabilities[0][predicted].item()
                
                return predicted.item(), confidence, landmarks
    
    # Special case for "nothing" gesture
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    if std_dev < 30:  # Threshold for uniform image
        return 23, 1.0, None  # Index for "nothing" with high confidence
    
    return None, 0.0, None

def draw_hand_landmarks(image, hand_landmarks):
    """Draw hand landmarks on the image"""
    if hand_landmarks:
        h, w, _ = image.shape
        pink_color = (255, 192, 203)  # Pink color in BGR
        
        # Draw landmarks as pink dots
        for landmark in hand_landmarks:
            x, y, z = landmark
            cx, cy = int(x * w), int(y * h)
            cv2.circle(image, (cx, cy), 5, pink_color, -1)
        
        # Define hand connections (finger joints)
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky finger
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        # Draw connections between landmarks
        for start_idx, end_idx in connections:
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                start_x, start_y, _ = hand_landmarks[start_idx]
                end_x, end_y, _ = hand_landmarks[end_idx]
                start_point = (int(start_x * w), int(start_y * h))
                end_point = (int(end_x * w), int(end_y * h))
                cv2.line(image, start_point, end_point, pink_color, 2)
    
    return image

def main():
    # Load the model and label mapping
    model, device = load_model()
    
    # Initialize hand detector
    detector = HandDetector(maxHands=1, detectionCon=0.7, trackCon=0.7)
    
    # Define label mapping
    allowed_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                      'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'NOTHING']
    label_map = {idx: letter for idx, letter in enumerate(allowed_letters)}
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize variables for smooth prediction display
    last_prediction = None
    last_confidence = 0.0
    prediction_time = 0
    prediction_duration = 0.5  # seconds to show prediction
    
    print("\nStarting real-time sign language recognition...")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Get current time
        current_time = time.time()
        
        # Only process every few frames to maintain performance
        if current_time - prediction_time >= 0.1:  # 10 FPS for prediction
            predicted_idx, confidence, hand_landmarks = predict_gesture(model, frame, device, detector)
            
            if predicted_idx is not None:
                last_prediction = label_map[predicted_idx]
                last_confidence = confidence
                prediction_time = current_time
        
        # Draw hand landmarks
        if hand_landmarks:
            frame = draw_hand_landmarks(frame, hand_landmarks)
        
        # Display prediction if it's still valid
        if last_prediction and (current_time - prediction_time) < prediction_duration:
            # Create a semi-transparent overlay for the prediction
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (300, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Add prediction text
            text = f"Predicted: {last_prediction} ({last_confidence:.2%})"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Sign Language Recognition', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
