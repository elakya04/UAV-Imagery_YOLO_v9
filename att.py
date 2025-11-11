import os
import torch
import pickle
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO

from captum.attr import GradientShap, IntegratedGradients, Occlusion, LayerGradCam
from captum.attr import visualization as viz

class YOLOAttentionExtractor:
    def _init_(self, model, target_layer_name=None, method='gradcam'):
        self.model = model
        self.model.eval()
        self.method = method
        self.target_layer = None
        if method == 'gradcam' and target_layer_name:
            for name, module in model.named_modules():
                if name == target_layer_name:
                    self.target_layer = module
                    break
            if self.target_layer is None:
                print(f"Warning: Target layer {target_layer_name} not found in model")

    def preprocess_yolo_output(self, output, image_size):
        if isinstance(output, list):
            max_conf = 0
            target_idx = 0
            for i, det in enumerate(output[0]):
                if det[4] > max_conf:
                    max_conf = det[4]
                    target_idx = i
            return output[0][target_idx][5]
        return output.sum()

    def get_attention_map(self, image):
        if not isinstance(image, torch.Tensor):
            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor()
            ])
            image = transform(image).unsqueeze(0)

        image = image.to(torch.float32)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.clone().requires_grad_(True)

        if self.method == 'gradcam':
            if self.target_layer is None:
                raise ValueError("Target layer not set for GradCAM")

            layer_gc = LayerGradCam(self.model, self.target_layer)

            def get_class_score(input_img):
                out = self.model(input_img)
                return self.preprocess_yolo_output(out, image.shape[2:])

            attribution = layer_gc.attribute(image, target=get_class_score)
            attribution = attribution.sum(dim=1, keepdim=True)
            attribution = torch.nn.functional.relu(attribution)
            attribution = attribution - attribution.min()
            if attribution.max() > 0:
                attribution = attribution / attribution.max()
            attention_map = torch.nn.functional.interpolate(
                attribution, size=image.shape[2:], mode='bilinear', align_corners=False)
            return attention_map.squeeze().cpu().numpy()

        elif self.method == 'integrated_gradients':
            ig = IntegratedGradients(self.model)
            baseline = torch.zeros_like(image)

            def get_class_score(input_img):
                out = self.model(input_img)
                return self.preprocess_yolo_output(out, image.shape[2:])

            attribution = ig.attribute(image, baseline, target=get_class_score, n_steps=50)
            attribution = attribution.sum(dim=1, keepdim=True)
            attribution = torch.abs(attribution)
            attribution = attribution - attribution.min()
            if attribution.max() > 0:
                attribution = attribution / attribution.max()
            return attribution.squeeze().cpu().numpy()

        elif self.method == 'occlusion':
            occlusion = Occlusion(self.model)

            def get_class_score(input_img):
                out = self.model(input_img)
                return self.preprocess_yolo_output(out, image.shape[2:])

            attribution = occlusion.attribute(
                image,
                target=get_class_score,
                sliding_window_shapes=(1, 16, 16),
                strides=(1, 8, 8)
            )
            attribution = attribution.sum(dim=1, keepdim=True)
            attribution = -attribution
            attribution = attribution - attribution.min()
            if attribution.max() > 0:
                attribution = attribution / attribution.max()
            if attribution.shape[2:] != image.shape[2:]:
                attention_map = torch.nn.functional.interpolate(
                    attribution, size=image.shape[2:], mode='bilinear', align_corners=False)
            else:
                attention_map = attribution
            return attention_map.squeeze().cpu().numpy()

def generate_attention_maps_for_dataset(model, image_dir, target_layer_name, method='gradcam', output_path='attention_maps.pkl'):
    extractor = YOLOAttentionExtractor(model, target_layer_name, method)

    attention_results = {}

    for filename in tqdm(os.listdir(image_dir), desc="Processing images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image_path = os.path.join(image_dir, filename)
                image = Image.open(image_path).convert('RGB')
                attention_map = extractor.get_attention_map(image)
                attention_results[filename] = attention_map
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    with open(output_path, 'wb') as f:
        pickle.dump(attention_results, f)

if __name__ == "_main_":
    model = YOLO("yolov9m.pt")
    image_dir = "./images"
    target_layer_name = "model.22"
    generate_attention_maps_for_dataset(model.model, image_dir, target_layer_name, method='gradcam', output_path='attention_maps.pkl')