import cv2
import numpy as np
import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from typing import Optional
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Portrait Cartoonify API", description="Convert portrait photos into cartoon-style images")

# Configuration constants
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB limit

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def validate_image(image_content: bytes) -> Optional[str]:
    """
    Validates image format and integrity.
    Returns error message if validation fails, None if successful.
    """
    if len(image_content) > MAX_IMAGE_SIZE:
        return f"Image too large (max {MAX_IMAGE_SIZE/1024/1024}MB allowed)"
    
    try:
        # Try opening with PIL to verify integrity
        img = Image.open(io.BytesIO(image_content))
        img.verify()  # Verify image integrity
        return None
    except Exception as e:
        return f"Invalid image file: {str(e)}"

def portrait_cartoonify(image_path: str, output_path: str, quality_level: str = "high") -> str:
    """
    Portrait cartoonization with skin tone preservation and enhanced quality.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output cartoon image
        quality_level: Processing quality - "low", "medium", or "high"
    
    Returns:
        Path to the processed cartoon image
    """
    logger.info(f"Processing image {image_path} with {quality_level} quality level")
    
    # Load image and check validity
    img = cv2.imread(image_path)
    if img is None or img.size == 0:
        raise ValueError("Could not load image or image is empty")
    
    # Store original dimensions
    original_height, original_width = img.shape[:2]
    logger.info(f"Original image dimensions: {original_width}x{original_height}")
    
    # Determine resize factor based on quality level
    if quality_level == "low":
        max_dimension = 800
    elif quality_level == "medium":
        max_dimension = 1200
    else:  # high
        max_dimension = 1600
    
    # Resize for processing if too large (improves speed and detection)
    scale = 1.0
    if max(original_height, original_width) > max_dimension:
        scale = max_dimension / max(original_height, original_width)
        img = cv2.resize(img, (int(original_width * scale), int(original_height * scale)))
        logger.info(f"Resized to {img.shape[1]}x{img.shape[0]} for processing")
    
    # Step 1: Enhanced Face detection
    faces = detect_faces(img)
    
    # Process based on face detection results
    if len(faces) == 0:
        logger.info("No faces detected, processing entire image")
        face_img = img.copy()
        is_face_only = False
    else:
        logger.info(f"Detected {len(faces)} faces, processing largest face")
        # Get the largest face with padding
        face_img, face_coords = extract_largest_face(img, faces)
        is_face_only = True
    
    # Step 2: Improved skin tone analysis
    skin_mask = detect_skin(face_img)
    
    # Step 3: Apply adaptive bilateral filtering
    smoothed = apply_adaptive_smoothing(face_img, skin_mask, quality_level)
    
    # Step 4: Enhanced edge detection
    edges = detect_edges(smoothed, quality_level)
    
    # Step 5: Color quantization with vibrant colors
    segmented_image = quantize_colors(smoothed, quality_level)
    
    # Step 6: Skin tone preservation
    segmented_image = preserve_skin_tone(segmented_image, face_img, skin_mask)
    
    # Step 7: Edge refinement and cartoon creation
    cartoon = apply_cartoon_effect(segmented_image, edges)
    
    # Step 8: Color enhancement and contrast improvement
    enhanced_cartoon = enhance_colors(cartoon)
    
    # Step 9: Final processing and background creation
    final_cartoon = final_processing(enhanced_cartoon)
    
    # Step 10: Composition and resizing
    result_img = compose_final_image(
        final_cartoon, 
        img, 
        original_width, 
        original_height, 
        is_face_only, 
        face_coords if is_face_only else None,
        quality_level
    )
    
    # Save result with appropriate quality
    if output_path.lower().endswith('.png'):
        cv2.imwrite(output_path, result_img)
    else:
        # For JPEG, set quality based on quality_level
        jpeg_quality = 95 if quality_level == "high" else (85 if quality_level == "medium" else 75)
        cv2.imwrite(output_path, result_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    
    logger.info(f"Successfully saved cartoon image to {output_path}")
    return output_path

def detect_faces(img):
    """Enhanced face detection with multiple cascade classifiers"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try to load face detection models, with fallbacks
    face_cascade_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
    ]
    
    # Try each face cascade until we find faces
    faces = []
    for cascade_path in face_cascade_paths:
        try:
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                logger.warning(f"Failed to load cascade classifier from {cascade_path}")
                continue
                
            # Try with different parameters
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                break
                
            faces = face_cascade.detectMultiScale(gray, 1.05, 3)
            if len(faces) > 0:
                break
        except Exception as e:
            logger.warning(f"Error with cascade {cascade_path}: {str(e)}")
    
    # Try profile face detection if still no face found
    if len(faces) == 0:
        try:
            profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            if not profile_cascade.empty():
                faces = profile_cascade.detectMultiScale(gray, 1.05, 3)
        except Exception as e:
            logger.warning(f"Error with profile cascade: {str(e)}")
    
    return faces

def extract_largest_face(img, faces):
    """Extract the largest face with padding"""
    # Get the largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    # Add padding around face (50% for better composition)
    padding_h = int(h * 0.5)
    padding_w = int(w * 0.5)
    
    # Calculate new coordinates with padding
    x_padded = max(0, x - padding_w)
    y_padded = max(0, y - padding_h)
    w_padded = min(img.shape[1] - x_padded, w + 2 * padding_w)
    h_padded = min(img.shape[0] - y_padded, h + 2 * padding_h)
    
    # Crop to face region with padding
    face_img = img[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
    face_coords = (x_padded, y_padded, w_padded, h_padded)
    
    return face_img, face_coords

def detect_skin(img):
    """Advanced skin detection using multiple color spaces"""
    # YCrCb for general skin detection
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower_skin_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    skin_mask1 = cv2.inRange(ycrcb_img, lower_skin_ycrcb, upper_skin_ycrcb)
    
    # HSV for additional skin tone detection (covers wider range of ethnicities)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Range for lighter skin tones
    lower_skin_hsv1 = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin_hsv1 = np.array([25, 160, 255], dtype=np.uint8)
    skin_mask2 = cv2.inRange(hsv_img, lower_skin_hsv1, upper_skin_hsv1)
    
    # Range for darker skin tones
    lower_skin_hsv2 = np.array([165, 20, 70], dtype=np.uint8)
    upper_skin_hsv2 = np.array([180, 160, 255], dtype=np.uint8)
    skin_mask3 = cv2.inRange(hsv_img, lower_skin_hsv2, upper_skin_hsv2)
    
    # Combine all skin masks
    skin_mask = cv2.bitwise_or(skin_mask1, cv2.bitwise_or(skin_mask2, skin_mask3))
    
    # Refine skin mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    
    return skin_mask

def apply_adaptive_smoothing(img, skin_mask, quality_level):
    """Apply adaptive bilateral filtering with different parameters for skin and non-skin areas"""
    # Parameters based on quality level
    if quality_level == "high":
        d_skin, sigma_color_skin, sigma_space_skin = 9, 15, 15
        d_other, sigma_color_other, sigma_space_other = 9, 30, 40
    elif quality_level == "medium":
        d_skin, sigma_color_skin, sigma_space_skin = 9, 20, 25
        d_other, sigma_color_other, sigma_space_other = 9, 40, 60
    else:  # low
        d_skin, sigma_color_skin, sigma_space_skin = 7, 30, 30
        d_other, sigma_color_other, sigma_space_other = 7, 50, 70
    
    # Apply different bilateral filter parameters to skin and non-skin regions
    smoothed_skin = cv2.bilateralFilter(img, d_skin, sigma_color_skin, sigma_space_skin)
    smoothed_other = cv2.bilateralFilter(img, d_other, sigma_color_other, sigma_space_other)
    
    # Combine based on skin mask
    smoothed = img.copy()
    for c in range(3):
        smoothed[:, :, c] = np.where(
            skin_mask > 0, 
            smoothed_skin[:, :, c], 
            smoothed_other[:, :, c]
        )
    
    return smoothed

def detect_edges(img, quality_level):
    """Enhanced edge detection with adaptive parameters"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur (parameters based on quality)
    if quality_level == "high":
        gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    else:
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholds based on image statistics
    median_val = np.median(gray_blurred)
    sigma = 0.33
    lower_threshold = int(max(0, (1.0 - sigma) * median_val))
    upper_threshold = int(min(255, (1.0 + sigma) * median_val))
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray_blurred, lower_threshold, upper_threshold)
    
    # Refine edges based on quality level
    if quality_level == "high":
        # Thinner, more precise edges for high quality
        kernel = np.ones((1, 1), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    else:
        # Thicker edges for low/medium quality
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges

def quantize_colors(img, quality_level):
    """Color quantization with adaptive number of clusters"""
    # Convert to LAB color space for better color segmentation
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Number of clusters based on quality level
    if quality_level == "high":
        k = 12  # More colors for high quality
    elif quality_level == "medium":
        k = 8   # Medium number of colors
    else:  # low
        k = 6   # Fewer colors for low quality
    
    # Reshape for k-means
    lab_pixels = lab_image.reshape((-1, 3))
    lab_pixels = np.float32(lab_pixels)
    
    # Apply k-means clustering with appropriate termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(lab_pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Convert back to 8-bit values and reshape
    centers = np.uint8(centers)
    segmented_lab = centers[labels.flatten()]
    segmented_lab = segmented_lab.reshape(img.shape)
    
    # Convert back to BGR
    segmented_image = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2BGR)
    
    return segmented_image

def preserve_skin_tone(segmented_img, original_img, skin_mask):
    """Preserve natural skin tones while maintaining cartoon effect"""
    # Adjust skin preservation level based on the overall brightness of skin areas
    skin_areas = cv2.bitwise_and(original_img, original_img, mask=skin_mask)
    if np.sum(skin_mask) > 0:  # Avoid division by zero
        avg_brightness = np.sum(cv2.cvtColor(skin_areas, cv2.COLOR_BGR2GRAY)) / np.sum(skin_mask)
        # Adjust preservation level based on brightness
        if avg_brightness > 150:
            skin_preservation = 0.2  # Less preservation for bright skin (cartoonish)
        else:
            skin_preservation = 0.4  # More preservation for darker skin tones
    else:
        skin_preservation = 0.3  # Default value
    
    # Apply weighted combination
    result = segmented_img.copy()
    for c in range(3):
        result[:, :, c] = np.where(
            skin_mask > 0, 
            cv2.addWeighted(original_img[:, :, c], skin_preservation, segmented_img[:, :, c], 1 - skin_preservation, 0),
            segmented_img[:, :, c]
        )
    
    return result

def apply_cartoon_effect(img, edges):
    """Apply cartoon effect by combining color segments with edges"""
    # Invert edges and convert to BGR
    edges_inv = cv2.bitwise_not(edges)
    edges_bgr = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)
    
    # Apply edge mask to the color-quantized image
    cartoon = cv2.bitwise_and(img, edges_bgr)
    
    return cartoon

def enhance_colors(img):
    """Enhance colors and contrast for a more vibrant cartoon look"""
    # Convert to HSV for better color manipulation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Increase saturation for more vibrant colors
    s = cv2.convertScaleAbs(s, alpha=1.25, beta=0)
    
    # Apply CLAHE to value channel for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    
    # Merge channels back
    hsv = cv2.merge((h, s, v))
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply subtle sharpening
    kernel_sharpen = np.array([[-0.1,-0.1,-0.1], [-0.1, 2.2,-0.1], [-0.1,-0.1,-0.1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)
    
    return enhanced

def final_processing(img):
    """Apply final processing steps and create clean white background"""
    # Create pure white background
    white_bg = np.ones(img.shape, dtype=np.uint8) * 255
    
    # Create mask from the cartoon
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    
    # Refine mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Convert mask to float for weighted addition
    mask = mask / 255.0
    
    # Apply mask to blend cartoon with white background
    result = img.copy()
    for c in range(3):
        result[:, :, c] = (1 - mask) * white_bg[:, :, c] + mask * img[:, :, c]
    
    return result

def compose_final_image(face_cartoon, original_img, original_width, original_height, is_face_only, face_coords, quality_level):
    """Compose the final image by either returning the whole cartoon or blending face with background"""
    # If entire image was processed
    if not is_face_only:
        return cv2.resize(face_cartoon, (original_width, original_height))
    
    # For face-only processing, we need to blend the cartoon face back into the original
    x, y, w, h = face_coords
    
    # Resize original image to original dimensions
    result_img = cv2.resize(original_img, (original_width, original_height))
    
    # Apply cartoon effect to the whole image (with simplified processing for background)
    result_img = apply_background_cartoonization(result_img, quality_level)
    
    # Calculate scaling factors to map face coordinates back to original size
    scale_x = original_width / original_img.shape[1]
    scale_y = original_height / original_img.shape[0]
    
    # Calculate face position in original scale
    x_orig = int(x * scale_x)
    y_orig = int(y * scale_y)
    w_orig = int(w * scale_x)
    h_orig = int(h * scale_y)
    
    # Resize cartoon face to match the face region in original image
    face_cartoon_resized = cv2.resize(face_cartoon, (w_orig, h_orig))
    
    # Create a mask for smooth blending
    blend_mask = create_blend_mask(h_orig, w_orig)
    
    # Apply the blended face to the result
    for c in range(3):
        result_img[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig, c] = (
            face_cartoon_resized[:, :, c] * blend_mask + 
            result_img[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig, c] * (1 - blend_mask)
        )
    
    return result_img

def apply_background_cartoonization(img, quality_level):
    """Apply simplified cartoon effect to background"""
    # Apply bilateral filter for smoothing
    if quality_level == "high":
        smoothed = cv2.bilateralFilter(img, 9, 75, 75)
    else:
        smoothed = cv2.bilateralFilter(img, 7, 50, 50)
    
    # Apply color quantization
    k = 8 if quality_level == "high" else 6
    
    # Convert to LAB
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    lab_pixels = lab.reshape((-1, 3))
    lab_pixels = np.float32(lab_pixels)
    
    # Apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(lab_pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Convert back
    centers = np.uint8(centers)
    segmented_lab = centers[labels.flatten()]
    segmented_lab = segmented_lab.reshape(img.shape)
    
    # Convert to BGR
    return cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2BGR)

def create_blend_mask(height, width):
    """Create a mask for smooth blending of face region with feathered edges"""
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Create feathered edges (gradient from 1 in center to 0 at edges)
    center_y, center_x = height // 2, width // 2
    max_dist = max(height, width) / 2
    
    # Create radial gradient
    y_coords, x_coords = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    
    # Normalize and invert distances, with smooth falloff
    normalized_dist = dist_from_center / max_dist
    mask = np.clip(1.0 - normalized_dist, 0.0, 1.0)
    
    # Apply sigmoid function for smoother transition
    mask = 1.0 / (1.0 + np.exp((normalized_dist - 0.7) * 10))
    
    return mask

def cleanup_old_files(directory, max_age_days=1):
    """Clean up old files from the specified directory"""
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {str(e)}")

@app.post("/cartoonify")
async def upload_file(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    quality: str = "high"  # Options: low, medium, high
):
    """
    Process an uploaded image and convert it to a cartoon style.
    
    Args:
        image: The image file to process
        quality: Processing quality (low, medium, high)
    
    Returns:
        The processed cartoon image
    """
    # Validate quality parameter
    if quality not in ["low", "medium", "high"]:
        quality = "high"  # Default to high if invalid
    
    # Validate file is an image
    content_type = image.content_type
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file extension
    file_ext = os.path.splitext(image.filename)[1].lower()
    if file_ext[1:] not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Read file content
    image_content = await image.read()
    
    # Validate image content
    error_msg = validate_image(image_content)
    if error_msg:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Create unique filename to prevent overwrites
    safe_filename = f"{int(time.time())}_{image.filename}"
    input_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    output_path = os.path.join(PROCESSED_FOLDER, f"cartoon_{safe_filename}")
    
    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(image_content)
    
    # Process the image
    try:
        logger.info(f"Processing image {input_path} with {quality} quality")
        result_path = portrait_cartoonify(input_path, output_path, quality)
        
        # Schedule cleanup of old files
        background_tasks.add_task(cleanup_old_files, UPLOAD_FOLDER)
        background_tasks.add_task(cleanup_old_files, PROCESSED_FOLDER)
        
        return FileResponse(
            result_path, 
            media_type="image/jpeg", 
            headers={"Content-Disposition": f"inline; filename=cartoon_{image.filename}"}
        )
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
@app.get("/")
def read_root():
    """API root endpoint with usage information"""
    return {
        "message": "Portrait Cartoonify API",
        "usage": "POST an image to /cartoonify endpoint",
        "parameters": {
            "image": "Image file to process",
            "quality": "Processing quality (low, medium, high)"
        }
    }

# Add missing imports at the top if you plan to use these functions
import time
