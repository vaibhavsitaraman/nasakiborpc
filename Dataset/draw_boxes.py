import cv2
import os
import matplotlib.pyplot as plt
import random
print("Hello World")
def plot_yolo_annotations(image_path, label_path, class_names, num_images_to_show=5):
    """
    Reads images and their YOLO annotations, then displays them with bounding boxes.

    Args:
        image_path (str): Path to the directory containing images.
        label_path (str): Path to the directory containing YOLO label files.
        class_names (list): A list of class names, where the index corresponds
                            to the class ID in the label files.
        num_images_to_show (int): Number of random images to display.
    """
    image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {image_path}")
        return

    random.shuffle(image_files) # Show a random sample

    for i, image_file in enumerate(image_files[:num_images_to_show]):
        img_full_path = os.path.join(image_path, image_file)
        label_file_name = os.path.splitext(image_file)[0] + ".txt"
        label_full_path = os.path.join(label_path, label_file_name)

        if not os.path.exists(label_full_path):
            print(f"Label file not found for {image_file}: {label_full_path}")
            continue

        # Read the image
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Could not read image: {img_full_path}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Matplotlib expects RGB
        h, w, _ = img.shape

        print(f"\nDisplaying: {image_file}")

        # Read YOLO annotations
        with open(label_full_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])

                # Convert YOLO format (normalized center x, center y, width, height)
                # to OpenCV format (absolute x_min, y_min, x_max, y_max)
                x_center_abs = x_center_norm * w
                y_center_abs = y_center_norm * h
                width_abs = width_norm * w
                height_abs = height_norm * h

                x_min = int(x_center_abs - (width_abs / 2))
                y_min = int(y_center_abs - (height_abs / 2))
                x_max = int(x_center_abs + (width_abs / 2))
                y_max = int(y_center_abs + (height_abs / 2))

                # Get class name
                try:
                    label_text = class_names[class_id]
                except IndexError:
                    label_text = f"ClassID:{class_id}"
                    print(f"Warning: Class ID {class_id} out of bounds for provided class_names.")


                # Draw bounding box
                cv2.rectangle(img_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) # Green box
                # Put class label
                cv2.putText(img_rgb, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2) # Blue text
                print(f"  - Found: {label_text} at [{x_min}, {y_min}, {x_max}, {y_max}]")


        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.title(image_file)
        plt.axis('off')
        plt.show()

        if i >= num_images_to_show -1:
            break

# --- Configuration ---
# Update these paths to match your dataset structure
dataset_root = '/Users/sambhargav/kiporpc/Dataset/output_yolo_dataset/'
train_images_path = os.path.join(dataset_root, 'images/train')
train_labels_path = os.path.join(dataset_root, 'labels/train')
# Or use 'val/images' and 'val/labels' to check validation set

class_names_from_yaml = ['emerald', 'key', 'shell', 'letter', 'fossil', 'crystal', 'diamond', 'compass', 'coral', 'coin', 'treasure_box']

# --- Run the plotting function ---
plot_yolo_annotations(train_images_path, train_labels_path, class_names_from_yaml, num_images_to_show=5)