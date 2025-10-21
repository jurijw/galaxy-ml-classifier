from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from classifier import GalaxyCNN


def load_image_tensor(img_path: Path) -> torch.Tensor:
    """
    Load an image from a path and transform it into a tensor suitable for the CNN.

    Args:
        img_path: Path to the input JPG image.

    Returns:
        Transformed image tensor of shape [C, H, W].
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img)


def visualize_feature_maps(model: torch.nn.Module,
                           img: torch.Tensor,
                           layer_name: str,
                           device: torch.device = torch.device("cpu"),
                           n_cols: int = 8) -> None:
    """
    Visualize feature maps from a convolutional layer of a PyTorch model.

    Args:
        model: The trained CNN model.
        img: Input image tensor of shape [C, H, W] (unsqueezed batch not needed).
        layer_name: Name of the layer to visualize ('conv1' or 'conv2').
        device: Device to run the model on (CPU or CUDA).
        n_cols: Number of columns in the plotted grid.
    """
    model.eval()
    features = {}

    # Hook to capture the output of the requested layer
    def hook_fn(module, input, output):
        features[module] = output.detach().cpu()

    layer = getattr(model, layer_name)
    handle = layer.register_forward_hook(hook_fn)

    # Forward pass (batch dimension required)
    img_batch = img.unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(img_batch)

    fmap = features[layer].squeeze(0)  # [num_channels, H, W]
    n_channels = fmap.shape[0]
    n_rows = (n_channels + n_cols - 1) // n_cols

    # Plot feature maps
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    for i in range(n_channels):
        fm = fmap[i]
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-5)  # normalize [0,1]
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(fm, cmap='gray')
        plt.axis('off')
        plt.title(f'Filter {i}')
    plt.tight_layout()
    plt.show()

    handle.remove()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = GalaxyCNN()
    model.load_state_dict(torch.load("galaxy_cnn.pth", map_location=device))
    model.to(device)
    model.eval()

    # Load example images
    elliptical_path = Path("images/elliptical/1237648720140501189.jpg")
    spiral_path = Path("images/spiral/1237645943979311221.jpg")

    tensor_elliptical = load_image_tensor(elliptical_path)
    tensor_spiral = load_image_tensor(spiral_path)

    # Visualize feature maps
    for layer in ['conv1', 'conv2']:
        visualize_feature_maps(model, tensor_elliptical, layer_name=layer, device=device)
        visualize_feature_maps(model, tensor_spiral, layer_name=layer, device=device)
