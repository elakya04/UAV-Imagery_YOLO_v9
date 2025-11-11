import os
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


class CustomGradCAM:
    def _init_(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.features = None
        self.hooks = []
        

        def forward_hook(module, input, output):

            if isinstance(output, (tuple, list)):
                self.features = output[0]
            else:
                self.features = output
            return None
        

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, (tuple, list)):
                self.gradients = grad_output[0]
            else:
                self.gradients = grad_output
            return None
        

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        

        self.hooks = [forward_handle, backward_handle]
        
    def _del_(self):

        for hook in self.hooks:
            hook.remove()
    
    def generate_attention_map(self, input_tensor, target_class=None):

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True
        

        self.gradients = None
        self.features = None
        
        try:

            with torch.set_grad_enabled(True):
                output = self.model(input_tensor)
                


                pred_tensor = None
                if isinstance(output, dict):


                    if 'pred' in output:
                        pred_tensor = output['pred']
                    elif 'pred_cls' in output:
                        pred_tensor = output['pred_cls']
                    else:

                        for k, v in output.items():
                            if isinstance(v, torch.Tensor):
                                pred_tensor = v
                                break
                elif isinstance(output, (list, tuple)):

                    if isinstance(output[0], torch.Tensor):
                        pred_tensor = output[0]
                    else:

                        for item in output:
                            if isinstance(item, torch.Tensor):
                                pred_tensor = item
                                break
                else:

                    pred_tensor = output
                

                if pred_tensor is None:

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

                    if target_class is None:

                        score = pred_tensor.sum()
                    else:

                        try:
                            score = pred_tensor[..., target_class].sum()
                        except IndexError:
                            score = pred_tensor.sum()
                

                self.model.zero_grad()
                score.backward(retain_graph=True)
            

            if self.gradients is None or self.features is None:
                print("Warning: Gradients or features weren't captured. Using fallback method.")
                return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32) * 0.5
            

            gradients = self.gradients.detach()
            features = self.features.detach()
            

            if gradients.dim() == 4:

                weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            elif gradients.dim() == 3:
                weights = torch.mean(gradients, dim=(1, 2), keepdim=True).unsqueeze(0)
            else:

                print(f"Warning: Unexpected gradient dimensions: {gradients.shape}. Using uniform weights.")
                weights = torch.ones_like(gradients.mean(dim=list(range(1, gradients.dim())), keepdim=True))
            

            if weights.dim() != features.dim():

                while weights.dim() < features.dim():
                    weights = weights.unsqueeze(-1)
                while weights.dim() > features.dim():
                    weights = weights.mean(-1)
            

            channel_dim = 1
            if features.shape[0] == input_tensor.shape[1] and features.dim() == 3:

                channel_dim = 0
                weights = weights.squeeze(0)
            

            if channel_dim == 1:
                attention_map = torch.sum(weights * features, dim=1, keepdim=True)
            else:
                attention_map = torch.sum(weights * features, dim=0, keepdim=True)
            

            attention_map = torch.relu(attention_map)
            

            attention_map = attention_map - attention_map.min()
            if attention_map.max() > 0:
                attention_map = attention_map / attention_map.max()
            

            while attention_map.dim() < 3:
                attention_map = attention_map.unsqueeze(0)
            

            attention_map = torch.nn.functional.interpolate(
                attention_map, 
                size=input_tensor.shape[2:],
                mode='bilinear', 
                align_corners=False
            )
            
            return attention_map.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"Error in generate_attention_map: {e}")

            return np.ones((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32) * 0.5


class YOLOAttentionExtractor:
    def _init_(self, model, target_layer_name):
        self.model = model
        self.model.eval()
        

        self.target_layer = None
        for name, module in model.named_modules():
            if name == target_layer_name:
                self.target_layer = module
                break
        
        if self.target_layer is None:
            raise ValueError(f"Layer {target_layer_name} not found in model")
            

        self.grad_cam = CustomGradCAM(self.model, self.target_layer)
        
    def preprocess_image(self, image_path):


        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        

        input_size = self.model.input_size if hasattr(self.model, 'input_size') else (640, 640)
        img_resized = cv2.resize(img, input_size)
        

        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        return img, img_tensor
    
    def get_attention_map(self, image_tensor):

        return self.grad_cam.generate_attention_map(image_tensor)


class AttentionGuidedAugmenter:
    def _init_(self, attention_threshold=0.7):
        self.attention_threshold = attention_threshold
        self.augmentation_types = ['noise', 'blur', 'color_shift', 'contrast']
        
    def get_high_attention_regions(self, attention_map):


        norm_map = (attention_map * 255).astype(np.uint8)
        

        _, binary_map = cv2.threshold(norm_map, int(self.attention_threshold * 255), 255, cv2.THRESH_BINARY)
        

        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 10:
                regions.append((x, y, w, h))
        
        return regions
    
    def apply_regional_augmentations(self, image, attention_map):

        regions = self.get_high_attention_regions(attention_map)
        augmented_image = image.copy()
        

        if not regions:
            aug_type = np.random.choice(self.augmentation_types)
            return self._apply_augmentation(augmented_image, aug_type, strength=0.5)
        
        for x, y, w, h in regions:

            aug_type = np.random.choice(self.augmentation_types)
            

            region = augmented_image[y:y+h, x:x+w]
            

            augmented_region = self._apply_augmentation(region, aug_type)
            

            augmented_image[y:y+h, x:x+w] = augmented_region
            
        return augmented_image
    
    def _apply_augmentation(self, img, aug_type, strength=1.0):

        if aug_type == 'noise':
            noise = np.random.normal(0, 25 * strength, img.shape).astype(np.uint8)
            return cv2.add(img, noise)
            
        elif aug_type == 'blur':
            kernel_size = max(1, int(5 * strength))
            if kernel_size % 2 == 0:
                kernel_size += 1
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
            
        return img


def save_visualizations(image, attention_map, augmented_image, output_path):


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    

    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    

    axes[1].imshow(attention_map, cmap='jet')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    

    axes[2].imshow(augmented_image)
    axes[2].set_title('Augmented Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def create_augmented_dataset(model, input_dir, output_dir, target_layer, 
                          attention_threshold=0.7, vis_dir=None):


    extractor = YOLOAttentionExtractor(model, target_layer)
    augmenter = AttentionGuidedAugmenter(attention_threshold)
    

    os.makedirs(output_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in Path(input_dir).glob('/*') 
                  if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    

    for img_path in tqdm(image_files):

        try:
            rel_path = img_path.relative_to(input_dir)
        except ValueError:

            rel_path = Path(os.path.basename(img_path))
            
        output_path = Path(output_dir) / rel_path
        

        os.makedirs(output_path.parent, exist_ok=True)
        
        try:

            if not os.path.isfile(img_path) or os.path.getsize(img_path) == 0:
                print(f"Skipping invalid file: {img_path}")
                continue
                

            try:
                original_img, img_tensor = extractor.preprocess_image(str(img_path))
            except Exception as e:
                print(f"Error preprocessing {img_path}: {e}")
                continue
                

            try:
                attention_map = extractor.get_attention_map(img_tensor)
                

                if attention_map is None or np.isnan(attention_map).any():
                    print(f"Invalid attention map for {img_path}, using fallback")
                    attention_map = np.ones((original_img.shape[0], original_img.shape[1]), dtype=np.float32) * 0.5
            except Exception as e:
                print(f"Error generating attention map for {img_path}: {e}")
                attention_map = np.ones((original_img.shape[0], original_img.shape[1]), dtype=np.float32) * 0.5
            

            augmented_img = augmenter.apply_regional_augmentations(original_img, attention_map)
            

            augmented_img_rgb = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), augmented_img_rgb)
            

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

    try:

        import ultralytics
        model = ultralytics.YOLO(model_path)
        model = model.model.to(device)
        return model
    except:

        try:
            model = torch.load(model_path, map_location=device)
            if hasattr(model, 'model'):

                model = model.model
            return model
        except:
            raise ValueError(f"Failed to load model from {model_path}")


if __name__== "_main_":
    parser = argparse.ArgumentParser(description="Create attention-guided augmented dataset")
    parser.add_argument("--model", type=str, default="./yolo11m.pt", help="Path to YOLO model weights")
    parser.add_argument("--input-dir", type=str, default="./images", help="Input image directory")
    parser.add_argument("--output-dir", type=str, default="./augumentations", help="Output directory for augmented images")
    parser.add_argument("--target-layer", type=str, default="model.23", help="Target layer for attention extraction")
    parser.add_argument("--threshold", type=float, default=0.7, help="Attention threshold for augmentation")
    parser.add_argument("--vis-dir", type=str, default="./visual", help="Directory to save visualizations (optional)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    
    args = parser.parse_args()
    

    print(f"Loading model from {args.model}")
    model = load_yolo_model(args.model, args.device)
    

    create_augmented_dataset(
        model=model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_layer=args.target_layer,
        attention_threshold=args.threshold,
        vis_dir=args.vis_dir
    )