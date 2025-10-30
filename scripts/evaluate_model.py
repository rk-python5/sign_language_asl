import torch
import cv2
import numpy as np
import mediapipe as mp
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from train_model import SignLanguageAttentionModel
import time

def enhance_image(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels
    limg = cv2.merge((cl,a,b))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced


def preprocess_image(image_path, save_debug=True):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    def enhance_image(img):
        # Convert BGR to RGB for MediaPipe
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image to a standard size (MediaPipe works better with square images)
        img = cv2.resize(img, (640, 640))
        
        # Convert to grayscale for preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        # Blend with original to maintain some color information
        alpha = 0.6  # More weight to enhanced image
        beta = 0.4   # Less weight to original
        final_image = cv2.addWeighted(enhanced, alpha, img, beta, 0)
        
        return final_image
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2,
        model_complexity=1
    )
    
    # Try different preprocessing variations
    landmarks = None
    detected_image = None
    
    for i in range(3):  # Try up to 3 different preprocessing variations
        if i == 0:
            processed_image = enhance_image(image)
        elif i == 1:
            # Try with original image
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_image = cv2.resize(processed_image, (640, 640))
        else:
            # Try with simple contrast enhancement
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_image = cv2.resize(processed_image, (640, 640))
            gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)
            processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Try different rotations if hand not detected
        rotations = [0, 90, -90, 180]
        for angle in rotations:
            if angle != 0:
                rows, cols = processed_image.shape[:2]
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                rotated = cv2.warpAffine(processed_image, M, (cols, rows))
                results = hands.process(rotated)
                
                if results.multi_hand_landmarks:
                    # Rotate back
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), -angle, 1)
                    detected_image = cv2.warpAffine(rotated, M, (cols, rows))
                    landmarks = results.multi_hand_landmarks[0].landmark
                    break
            else:
                results = hands.process(processed_image)
                if results.multi_hand_landmarks:
                    detected_image = processed_image.copy()
                    landmarks = results.multi_hand_landmarks[0].landmark
                    break
        
        if landmarks is not None:
            break
    
    if landmarks is None:
        return None
    
    if save_debug:
        # Save enhanced and detected images
        debug_dir = "dataset/debug_images"
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        
        # Save enhanced image
        enhanced = enhance_image(image)
        cv2.imwrite(os.path.join(debug_dir, f"enhanced_{base_name}"), cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        
        # Save detected image with landmarks
        if detected_image is not None:
            detected_image_bgr = cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    detected_image_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
            cv2.imwrite(os.path.join(debug_dir, f"detected_{base_name}"), detected_image_bgr)
    
    # Extract landmarks and normalize them
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Normalize coordinates to be relative to the wrist
    wrist = landmarks_array[0]
    landmarks_array = landmarks_array - wrist
    
    # Scale coordinates
    max_val = np.max(np.abs(landmarks_array))
    if max_val > 0:
        landmarks_array = landmarks_array / max_val
    
    return landmarks_array.flatten()

def evaluate_model(model_path, test_dir):
    # Create debug directory
    debug_dir = "dataset/debug_images"
    if os.path.exists(debug_dir):
        for f in os.listdir(debug_dir):
            os.remove(os.path.join(debug_dir, f))
    os.makedirs(debug_dir, exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageAttentionModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Initialize lists for predictions and true labels
    all_predictions = []
    all_labels = []
    processed_images = 0
    failed_detections = 0
    inference_times = []
    
    # Get all test images
    test_files = [f for f in os.listdir(test_dir) if f.endswith('_test.jpg')]
    
    print("\nProcessing test images...")
    
    # Process each test image
    for test_file in test_files:
        try:
            processed_images += 1
            # Get true label from filename
            label = test_file.split('_')[0]
            if label == 'nothing':
                label_idx = 23  # Assuming 'nothing' is the last class
            elif label == 'space':
                continue  # Skip space images
            else:
                label_idx = ord(label.upper()) - ord('A')
            
            # Preprocess image
            image_path = os.path.join(test_dir, test_file)
            landmarks = preprocess_image(image_path)
            
            if landmarks is not None:
                # Convert to tensor and make prediction
                landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Measure inference time
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else torch.zeros(1)
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else torch.zeros(1)
                
                if torch.cuda.is_available():
                    start_time.record()
                else:
                    start_time = torch.tensor([time.time()])
                
                with torch.no_grad():
                    outputs = model(landmarks_tensor)
                    _, predicted = torch.max(outputs, 1)
                
                if torch.cuda.is_available():
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
                else:
                    end_time = torch.tensor([time.time()])
                    inference_time = (end_time - start_time).item()
                
                inference_times.append(inference_time)
                all_predictions.append(predicted.item())
                all_labels.append(label_idx)
                print(f"✓ Successfully processed {test_file}")
            else:
                failed_detections += 1
                print(f"✗ No hand landmarks detected in {test_file}")
        
        except Exception as e:
            print(f"Error processing {test_file}: {str(e)}")
    
    print(f"\nProcessing complete:")
    print(f"Total images: {processed_images}")
    print(f"Successful detections: {len(all_predictions)}")
    print(f"Failed detections: {failed_detections}")
    print(f"\nDebug images saved in {debug_dir}/")
    
    if not all_predictions:
        print("\nNo valid predictions were made. Check if hands are being detected in the test images.")
        return
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    # Print performance metrics
    print("\nModel Performance Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Average Inference Time: {avg_inference_time:.4f} seconds")
    print(f"Inference Speed: {fps:.2f} FPS")
    
    # Generate detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_predictions))

if __name__ == "__main__":
    model_path = "sign_language_model.pth"
    test_dir = "dataset/test_set"
    evaluate_model(model_path, test_dir) 