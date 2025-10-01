from torchvision import transforms

def get_image_transform(image_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                            std=std)
    ])

    return transform
    

