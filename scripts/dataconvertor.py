import os
from typing import List, Tuple, Dict, Optional
import cv2
import mediapipe as mp
import numpy as np
import csv
import random


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Enhanced preprocessing for better hand detection.
    
    Args:
        image: Input BGR image as numpy array
        
    Returns:
        Preprocessed RGB image as numpy array
    """
    # Convert BGR to RGB for MediaPipe
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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


def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to the image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Augmented image as numpy array
    """
    # Random brightness adjustment
    if random.random() < 0.5:
        brightness = random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    # Random contrast adjustment
    if random.random() < 0.5:
        contrast = random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    
    # Random rotation (small angles)
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height))
    
    return image


def create_landmarks_dataset_single_folder(data_dir: str, output_filename: str) -> None:
    """
    Process images in all subfolders of 'data_dir' (A-Z).
    Extracts hand landmarks from images and saves to:
      - An NPZ file: output_filename.npz
      - A CSV file: output_filename.csv
      
    Args:
        data_dir: Directory containing subfolders of images
        output_filename: Base name for output files
    """
    data: List[List[float]] = []  # List to store 63-d landmark vectors
    labels: List[int] = []  # List to store numeric labels
    files: List[str] = []  # List to store full image paths

    # Define allowed letters (excluding 'J' and 'Z')
    allowed_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                      'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y']

    # Create a label mapping with consecutive indices
    label_map = {letter: idx for idx, letter in enumerate(allowed_letters)}

    # Iterate through all subfolders (A-Z)
    for subfolder in sorted(os.listdir(data_dir)):
        subfolder_path = os.path.join(data_dir, subfolder)

        if os.path.isdir(subfolder_path) and subfolder.upper() in label_map:  
            # Get full image paths instead of just filenames
            files.extend([os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print("Processing images and extracting hand landmarks...")

    # Initialize MediaPipe Hands with optimized parameters
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.2,  # Balanced threshold
        min_tracking_confidence=0.2,   # Balanced threshold
        model_complexity=1  # Use full model for better accuracy
    ) as hands:
        for image_path in files:
            filename = os.path.basename(image_path)
            label_letter = os.path.basename(os.path.dirname(image_path)).upper()

            if label_letter not in label_map:
                print(f"Skipping file {filename}: label '{label_letter}' is not allowed.")
                continue

            # Read and preprocess the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to load image {image_path}. Skipping...")
                continue
            
            # Try different preprocessing variations
            success = False
            for i in range(3):  # Try up to 3 different preprocessing variations
                if i == 0:
                    processed_image = preprocess_image(image)
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
                
                # Process with MediaPipe
                results = hands.process(processed_image)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    flat_landmarks = [coord for point in landmarks for coord in point]

                    if len(flat_landmarks) == 63:
                        data.append(flat_landmarks)
                        labels.append(label_map[label_letter])
                        print(f"Successfully processed {filename} with variation {i+1}")
                        success = True
                        break
            
            if not success:
                print(f"No hand detected in image {image_path} with any preprocessing variation")

    # Convert lists to numpy arrays
    data_np = np.array(data)
    labels_np = np.array(labels)

    # Save dataset to NPZ
    npz_filename = output_filename + ".npz"
    np.savez(npz_filename, data=data_np, labels=labels_np, label_map=label_map)
    print(f"Dataset saved to {npz_filename} with {len(data_np)} samples.")

    # Save dataset to CSV
    csv_filename = output_filename + ".csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["label"] + [f"lm_{i + 1}" for i in range(63)]
        writer.writerow(header)
        for lab, features in zip(labels_np, data_np):
            writer.writerow([lab] + list(features))
    print(f"Dataset saved to {csv_filename} with {len(data_np)} samples.")


if __name__ == '__main__':
    data_directory = "train_set"
    output_file = "landmarks_dataset"
    create_landmarks_dataset_single_folder(data_directory, output_file)