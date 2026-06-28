
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config


class DatasetLoader:
    """
    Loads and preprocesses CIFAR-10 or MNIST.
    Automatically downloads dataset to Config.DATA_DIR on first run.
    """

    def __init__(self, dataset_name=Config.DATASET):
        self.dataset_name = dataset_name
        self.train_loader = None
        self.test_loader  = None
        self._load()

    # ------------------------Preprocessing transforms----------------------------
    def _get_transforms(self):

        if self.dataset_name == 'CIFAR10':
            # Training: augmentation applied to reduce overfitting
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),       # pad 4px then random crop
                transforms.RandomHorizontalFlip(p=0.5),     # random mirror
                transforms.ToTensor(),                       # scale [0,255] → [0.0,1.0]
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),           # per-channel mean (R,G,B)
                    std=(0.2023, 0.1994, 0.2010)             # per-channel std  (R,G,B)
                )
            ])
            # Testing: NO augmentation — only normalize
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)
                )
            ])

        elif self.dataset_name == 'MNIST':
            # MNIST is grayscale and clean — no augmentation needed
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # single channel
            ])
            test_transform = train_transform  # same for MNIST

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. Use 'CIFAR10' or 'MNIST'.")

        return train_transform, test_transform

    # -------------------------Load dataset--------------------------
    def _load(self):
        train_transform, test_transform = self._get_transforms()

        if self.dataset_name == 'CIFAR10':
            train_data = datasets.CIFAR10(
                root=Config.DATA_DIR, train=True,
                download=True, transform=train_transform
            )
            test_data = datasets.CIFAR10(
                root=Config.DATA_DIR, train=False,
                download=True, transform=test_transform
            )

        elif self.dataset_name == 'MNIST':
            train_data = datasets.MNIST(
                root=Config.DATA_DIR, train=True,
                download=True, transform=train_transform
            )
            test_data = datasets.MNIST(
                root=Config.DATA_DIR, train=False,
                download=True, transform=test_transform
            )

        self.train_loader = DataLoader(
            train_data,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True        # speeds up GPU transfer
        )
        self.test_loader = DataLoader(
            test_data,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        print(f"Dataset     : {self.dataset_name}")
        print(f"Train size  : {len(train_data)}")
        print(f"Test size   : {len(test_data)}")
        print(f"Batch size  : {Config.BATCH_SIZE}")

    def get_loaders(self):
        return self.train_loader, self.test_loader