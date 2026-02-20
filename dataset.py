from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import random
import torchvision.transforms as transforms
from config import NUM_SAMPLES, IMAGE_SIZE


class ShapeDataset(Dataset):
    def __init__(self, n_samples=NUM_SAMPLES, image_size=IMAGE_SIZE, transform=None):
        self.n_samples = n_samples
        self.image_size = image_size
        self.transform = transform

        self.data = []
        self.labels = []

        for _ in range(n_samples):
            label = random.randint(0, 1)
            image = self._generate_image(label)

            self.data.append(image)
            self.labels.append(label)

    def _generate_image(self, label):
        img = Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        if label == 0:
            draw.rectangle([16, 16, 48, 48], fill=(255, 0, 0))
        else:
            draw.ellipse([16, 16, 48, 48], fill=(0, 0, 255))

        return img

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])