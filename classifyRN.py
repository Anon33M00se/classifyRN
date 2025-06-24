#
#  Use torchvision pre trained model ResNet (with default pretrained weights)
#

from PIL import Image
from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms.functional import normalize, resize, to_tensor, center_crop

def classify (
    image: Image.Image,
) -> str:
    """
    Classify an image using ResNet

    Parameters:
        image (PIL.Image): Input image.

    Returns:
        str: Best guess at classification of image.
    """

    #
    # Init ResNet models with default weights
    #
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Preprocess image
    image = image.convert("RGB")
    image = resize(image, 232)
    image = center_crop(image, 224)
    image_tensor = to_tensor(image)
    normalized_tensor = normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Generate predicted category
    prediction = model(normalized_tensor.unsqueeze(0)).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    return(category_name)


if __name__ == "__main__":
  #image = Image.open("test/assets/grace_hopper_517x606.jpg")
  image = Image.open("test/assets/moose.jpg")
  #image = Image.open("test/assets/ham_sandwich.jpg")
  print(f"   Input image: {image.filename}")
  print(f"Classification: {classify(image)}")
  
