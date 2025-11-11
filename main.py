import os
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


class CustomGradCAM:
    """Custom implementation of Grad-CAM without relying on captum library"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        # Register hooks for gradient and feature map capture
        self.gradients = None
        self.features = None
        self.hooks = []
        
        # Register forward hook to capture feature maps
        def forward_hook(module, input, output):
            # Handle output which could be a tensor or tuple/list
            if isinstance(output, (tuple, list)):
                self.features = output[0]  # Take the first element if it's a tuple/list
            else:
                self.features = output
            return None
        
        # Register backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, (tuple, list)):
                self.gradients = grad_output[0]
            else:
                self.gradients = grad_output
            return None
        
        # Register the hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        # Save the hook handles for later removal
        self.hooks = [forward_handle, backward_handle]
        
    def __del__(self):
        # Remove hooks when object is deleted
        for hook in self.hooks:
            hook.remove()
    
    def generate_attention_map(self, input_tensor, target_class=None):
        """Generate attention map for the input tensor"""
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True
        
        # Reset stored gradients and features
        self.gradients = None
        self.features = None
        
        try:
            # Forward pass
            with torch.set_grad_enabled(True):
                output = self.model(input_tensor)
                
                # For YOLO models, we need to extract the appropriate output
                # YOLO models typically output detection tensor or dictionary with multiple outputs
                pred_tensor = None
                if isinstance(output, dict):
                    # Get the tensor containing class predictions
                    # This might need to be adapted based on your specific YOLO implementation
                    if 'pred' in output:
                        pred_tensor = output['pred']
                    elif 'pred_cls' in output:
                        pred_tensor = output['pred_cls']
                    else:
                        # If structured output, use the first tensor-like object
                        for k, v in output.items():
                            if isinstance(v, torch.Tensor):
                                pred_tensor = v
                                break
                elif isinstance(output, (list, tuple)):
                    # Some YOLO implementations return a list/tuple of tensors
                    if isinstance(output[0], torch.Tensor):
                        pred_tensor = output[0]  # Usually the first element contains predictions
                    else:
                        # If nested structure, try to find a tensor
                        for item in output:
                            if isinstance(item, torch.Tensor):
                                pred_tensor = item
                                break
                else:
                    # Direct tensor output
                    pred_tensor = output
                
                # If we couldn't find a suitable tensor, just sum the whole output
                if pred_tensor is None:
                    # Try to flatten output to a single value if possible
                    score = torch.tensor(0.0, requires_grad=True, device=device)
                    if isinstance(output, torch.Tensor):
                        score = output.sum()
                    elif isinstance(output, (list, tuple)):
                        for item in output:
                            if isinstance(item, torch.Tensor):
                                score = score + item.sum()
                    elif isinstance(output, dict):
                        for k, v in output.items():
                            if isinstance(v, torch.Tensor):
                                score = score + v.sum()
                else:
                    # If target class not specified, use a general approach
                    if target_class is None:
                        # Just use the sum of all elements in the tensor
                        score = pred_tensor.sum()
                    else:
                        # Use provided target class if possible
                        try:
                            score = pred_tensor[..., target_class].sum()
                        except IndexError:
                            score = pred_tensor.sum()
                
                # Clear gradients and backward pass
                self.model.zero_grad()
                score.backward(retain_graph=True)
            
            # Check if we have captured the gradients and features
            if self.gradients is None or self.features is None:
                print("Warning: Gradients or features weren't captured. Using fallback method.")
                return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32) * 0.5
            
            # Get gradients and features
            gradients = self.gradients.detach()
            features = self.features.detach()
            
            # Handle different dimensions for global average pooling
            if gradients.dim() == 4:  # [B, C, H, W]
                # Standard case: [batch_size, channels, height, width]
                weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            elif gradients.dim() == 3:  # [C, H, W] - no batch dimension
                weights = torch.mean(gradients, dim=(1, 2), keepdim=True).unsqueeze(0)
            else:
                # Fallback for unusual dimensions
                print(f"Warning: Unexpected gradient dimensions: {gradients.shape}. Using uniform weights.")
                weights = torch.ones_like(gradients.mean(dim=list(range(1, gradients.dim())), keepdim=True))
            
            # Make sure dimensions match for multiplication
            if weights.dim() != features.dim():
                # Try to adapt weights to match feature dimensions
                while weights.dim() < features.dim():
                    weights = weights.unsqueeze(-1)
                while weights.dim() > features.dim():
                    weights = weights.mean(-1)
            
            # Handle different channel axis
            channel_dim = 1  # Default channel dimension (NCHW format)
            if features.shape[0] == input_tensor.shape[1] and features.dim() == 3:
                # [C, H, W] format
                channel_dim = 0
                weights = weights.squeeze(0)
            
            # Compute weighted combination of forward activation maps
            if channel_dim == 1:
                attention_map = torch.sum(weights * features, dim=1, keepdim=True)
            else:
                attention_map = torch.sum(weights * features, dim=0, keepdim=True)
            
            # Apply ReLU to focus on features that have a positive influence
            attention_map = torch.relu(attention_map)
            
            # Normalize to [0, 1]
            attention_map = attention_map - attention_map.min()
            if attention_map.max() > 0:
                attention_map = attention_map / attention_map.max()
            
            # Make sure attention_map has at least 3 dimensions for interpolate
            while attention_map.dim() < 3:
                attention_map = attention_map.unsqueeze(0)
            
            # Resize to input size
            attention_map = torch.nn.functional.interpolate(
                attention_map, 
                size=input_tensor.shape[2:],
                mode='bilinear', 
                align_corners=False
            )
            
            return attention_map.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"Error in generate_attention_map: {e}")
            # Return a fallback attention map
            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32) * 0.5


class YOLOAttentionExtractor:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()
        
        # Find the target layer in the model
        self.target_layer = None
        for name, module in model.named_modules():
            if name == target_layer_name:
                self.target_layer = module
                break
        
        if self.target_layer is None:
            raise ValueError(f"Layer {target_layer_name} not found in model")
            
        # Setup custom GradCAM
        self.grad_cam = CustomGradCAM(self.model, self.target_layer)
        
    def preprocess_image(self, image_path):
        """Load and preprocess an image for the model"""
        # Read image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        input_size = self.model.input_size if hasattr(self.model, 'input_size') else (640, 640)
        img_resized = cv2.resize(img, input_size)
        
        # Normalize and convert to tensor
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        return img, img_tensor
    
    def get_attention_map(self, image_tensor):
        """Generate attention map using custom GradCAM"""
        return self.grad_cam.generate_attention_map(image_tensor)


class AttentionGuidedAugmenter:
    def __init__(self, attention_threshold=0.7):
        self.attention_threshold = attention_threshold
        self.augmentation_types = ['noise', 'blur', 'color_shift', 'contrast']
        
    def get_high_attention_regions(self, attention_map):
        """Extract regions with high attention"""
        # Convert to uint8 for OpenCV processing
        norm_map = (attention_map * 255).astype(np.uint8)
        
        # Threshold the attention map to find high attention regions
        _, binary_map = cv2.threshold(norm_map, int(self.attention_threshold * 255), 255, cv2.THRESH_BINARY)
        
        # Find contours of high attention regions
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to bounding boxes
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:  # Filter out tiny regions
                regions.append((x, y, w, h))
        
        return regions
    
    def apply_regional_augmentations(self, image, attention_map):
        """Apply different augmentations to high-attention regions"""
        regions = self.get_high_attention_regions(attention_map)
        augmented_image = image.copy()
        
        # If no significant regions found, apply mild augmentation to whole image
        if not regions:
            aug_type = np.random.choice(self.augmentation_types)
            return self._apply_augmentation(augmented_image, aug_type, strength=0.5)
        
        for x, y, w, h in regions:
            # Select random augmentation for this region
            aug_type = np.random.choice(self.augmentation_types)
            
            # Extract region
            region = augmented_image[y:y+h, x:x+w]
            
            # Apply augmentation
            augmented_region = self._apply_augmentation(region, aug_type)
            
            # Put back the augmented region
            augmented_image[y:y+h, x:x+w] = augmented_region
            
        return augmented_image
    
    def _apply_augmentation(self, img, aug_type, strength=1.0):
        """Apply a specific augmentation to an image region"""
        if aug_type == 'noise':
            noise = np.random.normal(0, 25 * strength, img.shape).astype(np.uint8)
            return cv2.add(img, noise)
            
        elif aug_type == 'blur':
            kernel_size = max(1, int(5 * strength))
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size for GaussianBlur
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
        elif aug_type == 'color_shift':
            shift = np.array([
                np.random.randint(-50, 50),
                np.random.randint(-50, 50),
                np.random.randint(-50, 50)
            ]) * strength
            
            return np.clip(img.astype(np.float32) + shift, 0, 255).astype(np.uint8)
            
        elif aug_type == 'contrast':
            factor = 1.0 + np.random.uniform(-0.5, 0.5) * strength
            return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            
        return img  # Default: return original image


def save_visualizations(image, attention_map, augmented_image, output_path):
    """Save visualization of original, attention map, and augmented image"""
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    axes[1].imshow(attention_map, cmap='jet')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    
    # Augmented image
    axes[2].imshow(augmented_image)
    axes[2].set_title('Augmented Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def create_augmented_dataset(model, input_dir, output_dir, target_layer, 
                          attention_threshold=0.7, vis_dir=None):
    """Create and save an augmented dataset based on attention maps"""
    # Initialize extractor and augmenter
    extractor = YOLOAttentionExtractor(model, target_layer)
    augmenter = AttentionGuidedAugmenter(attention_threshold)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in Path(input_dir).glob('**/*') 
                  if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    for img_path in tqdm(image_files):
        # Get relative path for saving
        try:
            rel_path = img_path.relative_to(input_dir)
        except ValueError:
            # Handle case where the path is not relative
            rel_path = Path(os.path.basename(img_path))
            
        output_path = Path(output_dir) / rel_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        try:
            # Check if file exists and can be read
            if not os.path.isfile(img_path) or os.path.getsize(img_path) == 0:
                print(f"Skipping invalid file: {img_path}")
                continue
                
            # Load and preprocess image
            try:
                original_img, img_tensor = extractor.preprocess_image(str(img_path))
            except Exception as e:
                print(f"Error preprocessing {img_path}: {e}")
                continue
                
            # Get attention map with error handling
            try:
                attention_map = extractor.get_attention_map(img_tensor)
                
                # Verify attention map is valid
                if attention_map is None or np.isnan(attention_map).any():
                    print(f"Invalid attention map for {img_path}, using fallback")
                    attention_map = np.ones((original_img.shape[0], original_img.shape[1]), dtype=np.float32) * 0.5
            except Exception as e:
                print(f"Error generating attention map for {img_path}: {e}")
                attention_map = np.ones((original_img.shape[0], original_img.shape[1]), dtype=np.float32) * 0.5
            
            # Create augmented image
            augmented_img = augmenter.apply_regional_augmentations(original_img, attention_map)
            
            # Save augmented image
            augmented_img_rgb = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), augmented_img_rgb)
            
            # Save visualization if requested
            if vis_dir:
                vis_path = Path(vis_dir) / f"{rel_path.stem}_vis.png"
                os.makedirs(vis_path.parent, exist_ok=True)
                try:
                    save_visualizations(original_img, attention_map, augmented_img, str(vis_path))
                except Exception as e:
                    print(f"Error saving visualization for {img_path}: {e}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Augmented dataset saved to {output_dir}")
    if vis_dir:
        print(f"Visualizations saved to {vis_dir}")


def load_yolo_model(model_path, device='cuda'):
    """Load a YOLO model from path"""
    try:
        import ultralytics
        model = ultralytics.YOLO(model_path)
        model = model.model.to(device)
        return model
    except:
        # Fall back to torch.load for custom models
        try:
            model = torch.load(model_path, map_location=device)
            if hasattr(model, 'model'):
                # Sometimes the model is wrapped in a dict
                model = model.model
            return model
        except:
            raise ValueError(f"Failed to load model from {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create attention-guided augmented dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model weights")
    parser.add_argument("--input-dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for augmented images")
    parser.add_argument("--target-layer", type=str, default="model.23", help="Target layer for attention extraction")
    parser.add_argument("--threshold", type=float, default=0.7, help="Attention threshold for augmentation")
    parser.add_argument("--vis-dir", type=str, default=None, help="Directory to save visualizations (optional)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}")
    model = load_yolo_model(args.model, args.device)
    
    # Create augmented dataset
    create_augmented_dataset(
        model=model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_layer=args.target_layer,
        attention_threshold=args.threshold,
        vis_dir=args.vis_dir
    )