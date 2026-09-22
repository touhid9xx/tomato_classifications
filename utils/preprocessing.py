import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transform used for inference (no augmentation)
eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Transform used only for displaying the uploaded image at model input size
display_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])


def preprocess_for_model(pil_image: Image.Image) -> np.ndarray:
   
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    tensor = eval_transforms(pil_image)          # (3, 224, 224)
    batch = tensor.unsqueeze(0).numpy()          # type: ignore # (1, 3, 224, 224)
    return batch.astype(np.float32)


def preprocess_for_display(pil_image: Image.Image) -> Image.Image:
    
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return display_transforms(pil_image)