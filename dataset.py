import os
import cv2
import json # Kept for potential future COCO output, but not primary for this request
import random
import math
import numpy as np
from glob import glob
from tqdm import tqdm
import albumentations as A  # pip install albumentations
import yaml # For data.yaml

def inter_area(b1, b2):
    """Intersection area of two AABB boxes."""
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return (x1 - x0) * (y1 - y0)

def fits_box(new_box, existing_boxes, max_overlap_ratio):
    """Check if new_box overlaps too much with any existing box."""
    nx0, ny0, nx1, ny1 = new_box
    n_area = (nx1 - nx0) * (ny1 - ny0)
    if n_area == 0: return True # Avoid division by zero for zero-area boxes

    for b in existing_boxes:
        b_area = (b[2]-b[0])*(b[3]-b[1])
        if b_area == 0: continue # Skip zero-area existing boxes

        ia = inter_area(new_box, b)
        # Allow overlap up to max_overlap_ratio of the smaller box
        overlap_ratio = ia / min(n_area, b_area)
        if overlap_ratio > max_overlap_ratio:
            return False
    return True

def get_content_bounds(icon):
    """Get the actual bounding box of non-transparent content."""
    if icon.shape[2] == 4:  # Has alpha channel
        alpha = icon[:, :, 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
    else:
        # For 3-channel images, assume white/light pixels are background
        gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
        rows = np.any(gray < 240, axis=1)  # Non-white pixels
        cols = np.any(gray < 240, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return x_min, y_min, x_max + 1, y_max + 1

def degrade_image_realistic(image, line_variation=0.3):
    """
    Apply highly impactful pen-like degradation with extreme thickness variation and natural smoothing.
    Simulates dramatic pen pressure changes, ink bleeding, and realistic writing conditions.
    
    Args:
        image: Input BGR image
        line_variation: Amount of line thickness variation (0.0-1.0)
    
    Returns:
        Binary image (1 channel) with white objects on black background (no inversion)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Convert to binary (no inversion - keep original orientation)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    h, w = binary.shape
    result = binary.copy().astype(np.float32)
    
    # Create dramatic thickness variation map with extreme ranges
    thickness_map = np.ones((h, w), dtype=np.float32)
    
    # Create high-impact thickness variation patterns
    num_variation_layers = random.randint(20, 35)  # Much more frequent variations
    for layer in range(num_variation_layers):
        # Create random elliptical regions with varying intensities
        center_x = random.randint(-w//4, w + w//4)  # Allow off-canvas centers for edge effects
        center_y = random.randint(-h//4, h + h//4)
        
        # Much more dramatic size variations
        radius_x = random.randint(min(h,w)//25, min(h,w)//2)  # Wider range
        radius_y = random.randint(min(h,w)//30, min(h,w)//2)
        angle = random.uniform(0, 180)
        
        # Create smooth gradient mask with varying intensities
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Rotate coordinates for ellipse orientation
        cos_angle = np.cos(np.radians(angle))
        sin_angle = np.sin(np.radians(angle))
        x_rot = (x_coords - center_x) * cos_angle + (y_coords - center_y) * sin_angle
        y_rot = -(x_coords - center_x) * sin_angle + (y_coords - center_y) * cos_angle
        
        # Create elliptical gradient with varying falloff rates
        falloff_rate = random.uniform(1.0, 4.0)  # Variable gradient steepness
        ellipse_dist = (x_rot**2 / radius_x**2) + (y_rot**2 / radius_y**2)
        gradient_mask = np.exp(-ellipse_dist * falloff_rate)
        
        # Extreme thickness factors for dramatic variation
        thickness_factor = random.uniform(0.1, 3.5)  # Much wider range (10x thinner to 3.5x thicker)
        
        # Apply with higher impact
        impact_strength = 1.5 + line_variation  # Amplify the effect
        thickness_map *= (1.0 + (gradient_mask * (thickness_factor - 1.0) * impact_strength))
    
    # Apply extreme thickness variations
    # Create multiple erosion/dilation passes for dramatic effects
    processing_passes = random.randint(2, 4)
    
    for pass_idx in range(processing_passes):
        # Process in smaller chunks for more granular control
        chunk_size = random.randint(4, 12)
        
        for y in range(0, h, chunk_size):
            for x in range(0, w, chunk_size):
                chunk_h = min(chunk_size, h - y)
                chunk_w = min(chunk_size, w - x)
                
                # Get thickness for this chunk
                chunk_thickness = np.mean(thickness_map[y:y+chunk_h, x:x+chunk_w])
                chunk = result[y:y+chunk_h, x:x+chunk_w].astype(np.uint8)
                
                if chunk_thickness < 0.6:  # Very thin areas - aggressive erosion
                    erosion_strength = random.randint(2, 8)
                    kernel = np.ones((erosion_strength, erosion_strength), np.uint8)
                    chunk = cv2.erode(chunk, kernel, iterations=random.randint(1, 3))
                    
                elif chunk_thickness > 1.8:  # Very thick areas - dilation + blur
                    dilation_strength = random.randint(1, 5)
                    kernel = np.ones((dilation_strength, dilation_strength), np.uint8)
                    chunk = cv2.dilate(chunk, kernel, iterations=random.randint(1, 2))
                    
                    # Add ink bleeding effect for thick areas
                    if random.random() < 0.4:  # 40% chance of bleeding
                        bleeding_blur = random.randint(3, 9)
                        if bleeding_blur % 2 == 0: bleeding_blur += 1
                        chunk = cv2.GaussianBlur(chunk, (bleeding_blur, bleeding_blur), 1.5)
                        chunk = (chunk > 100).astype(np.uint8) * 255
                
                elif 0.8 < chunk_thickness < 1.4:  # Medium areas - subtle variation
                    if random.random() < 0.3:  # 30% chance of modification
                        kernel_size = random.randint(1, 3)
                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        if random.random() < 0.5:
                            chunk = cv2.erode(chunk, kernel, iterations=1)
                        else:
                            chunk = cv2.dilate(chunk, kernel, iterations=1)
                
                result[y:y+chunk_h, x:x+chunk_w] = chunk.astype(np.float32)
    
    # Apply dramatic multi-stage smoothing for natural pen effects
    result_smooth = result.copy()
    
    # Stage 1: Initial ink flow simulation
    flow_blur = random.randint(5, 9)  # Increased from (3, 7)
    if flow_blur % 2 == 0: flow_blur += 1
    result_smooth = cv2.GaussianBlur(result_smooth, (flow_blur, flow_blur), 1.2)  # Increased from 0.8
    
    # Stage 2: Pen texture simulation with directional blur
    pen_blur = random.randint(7, 13)  # Increased from (5, 11)
    if pen_blur % 2 == 0: pen_blur += 1
    result_smooth = cv2.GaussianBlur(result_smooth, (pen_blur, pen_blur), 2.4)  # Increased from 1.8
    
    # Stage 3: Paper absorption effect
    absorption_blur = max(9, int(25 * line_variation))  # Increased from max(7, int(20 * line_variation))
    if absorption_blur % 2 == 0: absorption_blur += 1
    absorption_smooth = cv2.GaussianBlur(result_smooth, (absorption_blur, absorption_blur), 4.0)  # Increased from 3.0
    
    # Stage 4: Ink settling effect
    settling_blur = max(11, int(30 * line_variation))  # Increased from max(9, int(25 * line_variation))
    if settling_blur % 2 == 0: settling_blur += 1
    settling_smooth = cv2.GaussianBlur(absorption_smooth, (settling_blur, settling_blur), 5.0)  # Increased from 4.0
    
    # Blend all smoothing stages for ultra-natural effect
    stage1_weight = 0.3
    stage2_weight = 0.25
    stage3_weight = 0.25 + 0.1 * line_variation
    stage4_weight = 0.2 + 0.15 * line_variation
    
    final_smooth = (
        result * stage1_weight +
        result_smooth * stage2_weight +
        absorption_smooth * stage3_weight +
        settling_smooth * stage4_weight
    )
    
    # Apply adaptive threshold with wider range
    base_threshold = 120 + random.randint(-30, 30)  # Much more variation
    threshold_variation = int(60 * line_variation)  # Wider range
    threshold_val = base_threshold - threshold_variation
    threshold_val = max(50, min(200, threshold_val))  # Clamp to reasonable range
    
    final_binary = (final_smooth > threshold_val).astype(np.uint8) * 255
    
    # Add random imperfections for realism
    if random.random() < 0.6:  # 60% chance of imperfections
        # Add small gaps and breaks
        break_kernel = np.ones((random.randint(1, 3), random.randint(1, 3)), np.uint8)
        if random.random() < 0.5:
            final_binary = cv2.erode(final_binary, break_kernel, iterations=1)
        
        # Add small blob connections
        if random.random() < 0.3:
            connect_kernel = np.ones((random.randint(2, 4), random.randint(2, 4)), np.uint8)
            final_binary = cv2.dilate(final_binary, connect_kernel, iterations=1)
    
    # Final natural edge smoothing with variable intensity
    final_edge_blur = random.randint(3, 8)  # Increased from (2, 6)
    if final_edge_blur % 2 == 0: final_edge_blur += 1
    final_binary = cv2.GaussianBlur(final_binary, (final_edge_blur, final_edge_blur), 1.0)  # Increased from 0.7
    
    # Final threshold with slight randomness
    final_threshold = random.randint(110, 140)
    final_binary = (final_binary > final_threshold).astype(np.uint8) * 255
    
    return final_binary
def generate_yolo_dataset(
    icons_dir: str,
    output_dir: str,
    num_images: int = 1000,
    img_size: tuple = (640, 640),
    icons_per_image: tuple = (5, 15),
    train_split: float = 0.8,
    max_place_attempts: int = 50,
    max_overlap_ratio: float = 0.4,
    margin: int = 10,
    # New degradation parameters
    apply_degradation: bool = True,
    line_variation: float = 0.3
):
    """
    Generates a synthetic dataset in YOLO format with realistic image degradation.
    """
    # Define YOLO directory structure
    train_images_dir = os.path.join(output_dir, 'images', 'train')
    val_images_dir = os.path.join(output_dir, 'images', 'val')
    train_labels_dir = os.path.join(output_dir, 'labels', 'train')
    val_labels_dir = os.path.join(output_dir, 'labels', 'val')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    icon_paths = sorted(glob(os.path.join(icons_dir, '*.png'))) # Sort for consistent class IDs
    if not icon_paths:
        raise RuntimeError(f"No icons found in {icons_dir}")

    # Class names for data.yaml (0-indexed)
    # The class ID in YOLO will be the index in this list
    class_names = [os.path.splitext(os.path.basename(p))[0] for p in icon_paths]

    aug = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.Perspective(scale=(0.02, 0.05), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    ])

    num_train_images = int(num_images * train_split)

    for img_idx_global in tqdm(range(num_images), desc="Generating images"):
        img_id_str = f"{img_idx_global:06d}" # Unique ID for file naming

        # Determine if this image is for training or validation
        if img_idx_global < num_train_images:
            current_image_dir = train_images_dir
            current_label_dir = train_labels_dir
        else:
            current_image_dir = val_images_dir
            current_label_dir = val_labels_dir

        canvas = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
        
        # These lists will store data *before* final augmentation for placement logic
        placed_icon_bbs_for_overlap_check = [] 
        # These lists will store data *for annotation* after final augmentation
        # The 'labels' will store the 0-indexed class ID
        final_masks_for_yolo = []
        final_class_ids_for_yolo = []

        n_icons = random.randint(*icons_per_image)
        
        # Generate icon choices with bias towards repeating first icon
        choices = [] # Stores the 0-indexed icon path index
        for i in range(n_icons):
            if i == 0 or len(icon_paths) == 1:
                choices.append(random.randrange(len(icon_paths)))
            else:
                if random.random() < 1/3:
                    choices.append(choices[0])
                else:
                    available_indices = [j for j in range(len(icon_paths)) if j != choices[0]]
                    if available_indices:
                        choices.append(random.choice(available_indices))
                    else: # Should not happen if len(icon_paths) > 1
                        choices.append(choices[0])
        
        temp_masks_for_augmentation = []

        for icon_original_idx in choices: # This is the 0-indexed class_id
            icon_path = icon_paths[icon_original_idx]
            icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
            if icon is None:
                print(f"Warning: Could not load icon {icon_path}")
                continue

            if icon.ndim == 2:
                icon = cv2.cvtColor(icon, cv2.COLOR_GRAY2BGRA)
            elif icon.shape[2] == 3:
                b, g, r = cv2.split(icon)
                a = np.ones_like(b) * 255
                icon = cv2.merge([b, g, r, a])
            
            alpha = icon[:, :, 3]
            binary_alpha = (alpha > 128).astype(np.uint8) * 255
            icon[:, :, 3] = binary_alpha

            h0, w0 = icon.shape[:2]
            ang = random.uniform(-180, 180)
            sc = random.uniform(0.27, 0.9)
            
            diag = int(math.sqrt(h0*h0 + w0*w0)) + 20
            large_canvas_icon = np.zeros((diag, diag, 4), dtype=np.uint8)
            
            start_y = (diag - h0) // 2
            start_x = (diag - w0) // 2
            large_canvas_icon[start_y:start_y+h0, start_x:start_x+w0] = icon
            
            M = cv2.getRotationMatrix2D((diag/2, diag/2), ang, sc)
            icon_t = cv2.warpAffine(large_canvas_icon, M, (diag, diag),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))
            
            content_bounds = get_content_bounds(icon_t)
            if content_bounds is None:
                continue
            
            x_min_cb, y_min_cb, x_max_cb, y_max_cb = content_bounds
            padding = 5
            x_min_cb = max(0, x_min_cb - padding)
            y_min_cb = max(0, y_min_cb - padding)
            x_max_cb = min(diag, x_max_cb + padding)
            y_max_cb = min(diag, y_max_cb + padding)
            
            icon_cropped = icon_t[y_min_cb:y_max_cb, x_min_cb:x_max_cb]
            ht, wt = icon_cropped.shape[:2]

            if wt <= 0 or ht <= 0: continue # Skip zero-size icons

            if wt > (img_size[0] - 2*margin) or ht > (img_size[1] - 2*margin):
                continue

            placed = False
            for _ in range(max_place_attempts):
                # x0, y0 are top-left coordinates on the main canvas
                x0 = random.randint(margin, img_size[0] - wt - margin)
                y0 = random.randint(margin, img_size[1] - ht - margin)
                # new_box_for_overlap_check is (xmin, ymin, xmax, ymax)
                new_box_for_overlap_check = (x0, y0, x0 + wt, y0 + ht)
                
                if fits_box(new_box_for_overlap_check, placed_icon_bbs_for_overlap_check, max_overlap_ratio):
                    placed_icon_bbs_for_overlap_check.append(new_box_for_overlap_check)
                    
                    # Composite onto canvas
                    alpha_mask_icon = icon_cropped[:, :, 3] > 0
                    canvas_section = canvas[y0:y0+ht, x0:x0+wt]
                    for c_channel in range(3): # BGR channels
                        canvas_section[alpha_mask_icon, c_channel] = icon_cropped[alpha_mask_icon, c_channel]
                    
                    # Create a mask for this specific icon on the full canvas size
                    # This mask will be augmented along with the image
                    icon_mask_on_canvas = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
                    icon_mask_on_canvas[y0:y0+ht, x0:x0+wt][alpha_mask_icon] = 255
                    temp_masks_for_augmentation.append(icon_mask_on_canvas)
                    
                    # Store the 0-indexed class ID for this mask
                    final_class_ids_for_yolo.append(icon_original_idx)
                    placed = True
                    break
            # if not placed:
            #     # print(f"Could not place icon {icon_original_idx}")
            #     pass # Continue to next icon if placement fails

        # Apply final augmentations to the entire canvas and all collected masks
        if temp_masks_for_augmentation: # Only augment if there are masks
            augmented = aug(image=canvas, masks=temp_masks_for_augmentation)
            canvas_augmented = augmented['image']
            final_masks_for_yolo = augmented['masks'] # These are the masks to use for YOLO BBoxes
        else: # No icons placed, or all failed
            canvas_augmented = canvas # Use original canvas
            final_masks_for_yolo = [] # No masks

        # Add subtle noise (original noise)
        noise = np.random.normal(0, 3, canvas_augmented.shape).astype(np.int16)
        canvas_with_noise = np.clip(canvas_augmented.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # NEW: Apply realistic degradation to simulate real-world conditions
        if apply_degradation:
            canvas_final = degrade_image_realistic(
                canvas_with_noise,
                line_variation=line_variation
            )
            # Convert binary image to BGR for consistency with the rest of the pipeline
            canvas_final = cv2.cvtColor(canvas_final, cv2.COLOR_GRAY2BGR)
        else:
            canvas_final = canvas_with_noise

        # Save image
        img_filename = f"{img_id_str}.png"
        cv2.imwrite(os.path.join(current_image_dir, img_filename), canvas_final)

        # Generate YOLO annotations
        yolo_annotations = []
        img_h, img_w = canvas_final.shape[:2]

        # Iterate through the *augmented* masks and their corresponding class IDs
        for i, mask_augmented in enumerate(final_masks_for_yolo):
            class_id = final_class_ids_for_yolo[i] # Get the stored 0-indexed class ID
            
            contours, _ = cv2.findContours(mask_augmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 10: # Min area threshold
                    continue
                
                # Get bounding box from contour
                x, y, w, h = cv2.boundingRect(cnt)

                # Convert to YOLO format (normalized and center x, center y, width, height)
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                # Clamp values to be within [0.0, 1.0] to avoid issues with tiny boxes near edges
                x_center = np.clip(x_center, 0.0, 1.0)
                y_center = np.clip(y_center, 0.0, 1.0)
                norm_w = np.clip(norm_w, 0.0, 1.0)
                norm_h = np.clip(norm_h, 0.0, 1.0)
                
                # Ensure width and height are not zero after clipping
                if norm_w == 0 or norm_h == 0:
                    continue

                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        # Save YOLO annotations to .txt file
        label_filename = f"{img_id_str}.txt"
        with open(os.path.join(current_label_dir, label_filename), 'w') as f:
            f.write("\n".join(yolo_annotations))

    # Create data.yaml for Ultralytics YOLO
    data_yaml_content = {
        'path': os.path.abspath(output_dir), # Or relative: ../Dataset/output_dataset
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml_content, f, sort_keys=False, default_flow_style=False)

    print(f"YOLO dataset generation completed in '{output_dir}'!")
    print(f"Generated {num_train_images} training images and {num_images - num_train_images} validation images.")
    print(f"Class names (0-indexed): {class_names}")
    print(f"data.yaml created at: {os.path.join(output_dir, 'data.yaml')}")
    if apply_degradation:
        print(f"Applied realistic degradation with line thickness variation: {line_variation}")


# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy icons for testing if you don't have them
    dummy_icons_dir = 'Dataset/template_icons_dummy'
    os.makedirs(dummy_icons_dir, exist_ok=True)
    
    # Create a few dummy PNG icons with alpha
    def create_dummy_icon(path, size, color, shape='rect'):
        img = np.zeros((size, size, 4), dtype=np.uint8) # BGRA
        if shape == 'rect':
            cv2.rectangle(img, (size//4, size//4), (size*3//4, size*3//4), color + (255,), -1)
        elif shape == 'circle':
            cv2.circle(img, (size//2, size//2), size//3, color + (255,), -1)
        cv2.imwrite(path, img)

    if not glob(os.path.join(dummy_icons_dir, '*.png')):
        print("Creating dummy icons...")
        create_dummy_icon(os.path.join(dummy_icons_dir, 'icon_A_square.png'), 100, (255,0,0), 'rect') # Blue
        create_dummy_icon(os.path.join(dummy_icons_dir, 'icon_B_circle.png'), 120, (0,255,0), 'circle') # Green
        create_dummy_icon(os.path.join(dummy_icons_dir, 'icon_C_triangle.png'), 80, (0,0,255), 'rect') # Red (using rect as placeholder)
        print("Dummy icons created.")

    generate_yolo_dataset(
        icons_dir= 'Dataset/template_icons', 
        output_dir='Dataset/output_yolo_dataset',
        num_images=800,  # Keep low for quick testing
        img_size=(640, 640),
        icons_per_image=(2, 5),
        train_split=0.8,
        max_overlap_ratio=0.3,
        margin=15,
        # NEW degradation parameters
        apply_degradation=True,       # Enable/disable degradation
        line_variation=0.3      # Amount of line thickness variation (0.0-1.0)
    )