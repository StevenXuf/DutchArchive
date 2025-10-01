import yaml

from torchvision import transforms

def get_default_config(config_path='./config.yaml'):
    with open(config_path,'r') as f:
        return yaml.safe_load(f)

def get_image_transform(image_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                            std=std)
    ])

    return transform
    

