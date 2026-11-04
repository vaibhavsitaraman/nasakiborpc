from ultralytics import YOLO
import torch # Optional: to check for GPU

def main():
    # Check for GPU
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")

    # --- Configuration ---
    # Path to your data.yaml file
    # Option 1: If data.yaml is in the same directory as your script or a known relative path
    # data_yaml_path = 'data.yaml'
    # Option 2: Absolute path (often more robust if script/data are in different places)
    data_yaml_path = '/Users/sambhargav/kiporpc/Dataset/output_dataset/yolo/data.yaml'

    # Choose a YOLOv8 model (e.g., yolov8n.pt for nano, yolov8s.pt for small)
    # Nano is fastest to train, good for starting, especially on CPU
    # Small offers a better balance if you have a GPU or more time
    model_variant = 'yolov8n.pt' # Start with nano, especially if on CPU
    # model_variant = 'yolov8s.pt'

    # Training parameters
    epochs = 100         # Number of training epochs (e.g., 50, 100, 200)
    img_size = 640      # Input image size
    batch_size = 16     # Adjust based on your GPU memory (if CPU, try 4, 8, or 16)
                        # If you get Out Of Memory (OOM) errors, reduce this
    patience_val = 20   # Early stopping patience: epochs to wait for improvement before stopping
    project_name = 'icon_detection_project' # Folder where results will be saved
    run_name = 'yolov8n_run1'              # Subfolder for this specific run

    # --- Load Model ---
    # Load a pre-trained YOLOv8 model
    model = YOLO(model_variant)

    # --- Train Model ---
    print(f"Starting training with model: {model_variant}")
    print(f"Dataset YAML: {data_yaml_path}")
    print(f"Epochs: {epochs}, Image Size: {img_size}, Batch Size: {batch_size}")

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=patience_val, # For early stopping
        device=device,         # 'cpu' or 'cuda' (or specific GPU like 0)
        project=project_name,  # Directory for saving runs
        name=run_name,         # Name of the training run
        # Other augmentations are on by default, you can customize if needed:
        # degrees=15,  # image rotation (+/- deg)
        # translate=0.1, # image translation (+/- fraction)
        # scale=0.5,   # image scale (+/- gain)
        # flipud=0.5,  # image flip up-down (probability)
    )

    print("Training complete!")
    print(f"Results saved to: {results.save_dir}")
    print(f"Best model weights saved at: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()