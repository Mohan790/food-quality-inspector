"""
gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM)

What is Grad-CAM? (explain this in your oral defense!)
    It creates a heatmap showing WHICH parts of the food image
    the model focused on to make its Fresh/Rotten decision.

    Example: For a rotten apple, the heatmap should highlight
    the brown/moldy spots — proving the model learned correctly!

Algorithm:
    1. Run forward pass → get prediction
    2. Backpropagate the predicted class score to last conv layer
    3. Average the gradients → importance weight per channel
    4. Weighted sum of feature maps + ReLU → heatmap
    5. Resize to 224x224 and overlay on original image
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Grad-CAM for FoodQualityClassifier.
    Hooks into the last conv block of MobileNetV2 features.
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

        # Storage filled by hooks during forward/backward pass
        self._gradients   = None
        self._activations = None

        # Hook into last feature block (highest-level features)
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        """Stores feature maps during forward pass."""
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Stores gradients during backward pass."""
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Generates Grad-CAM heatmap.

        Args:
            input_tensor: (1, 3, 224, 224) preprocessed image tensor
            class_idx: Target class (None = use predicted class)

        Returns:
            heatmap (np.ndarray): Normalized heatmap in [0,1]
            class_idx (int): The class that was explained
        """
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for target class only
        target_score = logits[0, class_idx]
        target_score.backward()

        # Compute channel importance weights
        # Global average pool gradients over spatial dims
        gradients   = self._gradients    # (1, C, H, W)
        activations = self._activations  # (1, C, H, W)
        weights     = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of feature maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Keep only positive contributions

        # Resize and normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, class_idx


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> Image.Image:
    """
    Overlays Grad-CAM heatmap on the original image.

    Colors:
        🔴 Red/Hot   = regions model focused on (suspicious areas)
        🔵 Blue/Cool = regions model ignored

    Args:
        original_image: PIL Image
        heatmap: np.ndarray in [0,1], shape (224, 224)
        alpha: Blend factor (0=original only, 1=heatmap only)

    Returns:
        PIL Image with heatmap overlay
    """
    img = np.array(original_image.resize((224, 224))).astype(np.float32)

    # Apply jet colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Blend original + heatmap
    overlay = (1 - alpha) * img + alpha * heatmap_color
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)