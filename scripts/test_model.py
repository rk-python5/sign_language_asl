import cv2
import mediapipe as mp
import torch
import numpy as np
from typing import Tuple, Dict, List
import os

from train_model import SignLanguageAttentionModel
from dataconvertor import preprocess_image


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Apply advanced contrast enhancement to an image.
    
    Args:
        image: Input RGB image as numpy array
        
    Returns:
        Enhanced image as numpy array
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def detect_skin(image: np.ndarray) -> np.ndarray:
    """
    Detect skin color in various color spaces.
    
    Args:
        image: Input BGR image as numpy array
        
    Returns:
        Image with only skin regions visible
    """
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


def load_model(model_path: str = 'sign_language_model.pth') -> Tuple[SignLanguageAttentionModel, torch.device]:
    """
    Load the trained model.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Tuple of (model, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageAttentionModel(num_classes=24).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def predict_gesture(model: SignLanguageAttentionModel, image_path: str, device: torch.device) -> Tuple[int, float]:
    """
    Predict the gesture from an image.
    
    Args:
        model: Trained model
        image_path: Path to the input image
        device: Device to run the model on
        
    Returns:
        Tuple of (predicted class index, confidence score)
        
    Raises:
        ValueError: If image cannot be loaded or no hand detected
    """
    # Read and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")
    
    # Try different preprocessing variations
    success = False
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
    
    # Try each variation with different MediaPipe configurations
    mp_configs = [
        {"min_detection_confidence": 0.2, "model_complexity": 1},
        {"min_detection_confidence": 0.1, "model_complexity": 1},
        {"min_detection_confidence": 0.2, "model_complexity": 0},
    ]
    
    for processed_image in variations:
        for config in mp_configs:
            # Initialize MediaPipe Hands with current config
            mp_hands = mp.solutions.hands
            with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=config["min_detection_confidence"],
                min_tracking_confidence=0.2,
                model_complexity=config["model_complexity"]
            ) as hands:
                # Convert to RGB for MediaPipe
                rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_image)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
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
                        
                        success = True
                        return predicted.item(), confidence
    
    # Special case for "nothing" gesture
    if not success:
        # Check if the image is mostly empty/uniform
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        if std_dev < 30:  # Threshold for uniform image
            return 23, 1.0  # Index for "nothing" with high confidence
    
    raise ValueError("No hand detected in the image")


def main() -> None:
    """Main function to test the model on a directory of images."""
    # Load the model and label mapping
    model, device = load_model()
    
    # Define label mapping (same as in training)
    allowed_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                      'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'NOTHING']
    label_map = {idx: letter for idx, letter in enumerate(allowed_letters)}
    
    # Test directory containing images
    test_dir = "dataset/test_set"
    
    # Process each image in the test directory
    print("\nTesting images...")
    for image_name in sorted(os.listdir(test_dir)):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, image_name)
            try:
                # Extract true label from filename
                true_letter = image_name.split('_')[0].upper()
                
                # Skip J, Z, and SPACE as they're not in our model
                if true_letter in ['J', 'Z', 'SPACE']:
                    print(f"\nSkipping {image_name}: {true_letter} not in model")
                    continue
                
                predicted_idx, confidence = predict_gesture(model, image_path, device)
                predicted_letter = label_map[predicted_idx]
                
                print(f"\nImage: {image_name}")
                print(f"True Label: {true_letter}")
                print(f"Predicted: {predicted_letter}")
                print(f"Confidence: {confidence:.2%}")
                print("-" * 50)
                
            except Exception as e:
                print(f"\nError processing {image_name}: {str(e)}")


if __name__ == "__main__":
    main() 