import torchvision

def get_transforms(aug=False):
    
    def transforms(img):
        img = img.convert('RGB') 
        if aug:
            tfm = [
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.RandomRotation(degrees=(-5, 5)), 
                torchvision.transforms.RandomResizedCrop((128, 128), scale=(0.8, 1), ratio=(0.45, 0.55)) 
            ]
            

        else:
            tfm = [
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.Resize((128, 128))
                

            ]

        # Normalize 
        img = torchvision.transforms.Compose(tfm + [            
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.2179, std=0.0529),
            
        ])(img)
        # print(f"img size in aug = True in transfomrm {img.shape}")
        return img

    return lambda img: transforms(img)