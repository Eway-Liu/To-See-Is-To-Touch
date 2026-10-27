"""Compatibility entry point; trainers now produce the same checkpoint."""
from .legacy_model import UNet
from .dataset import TactileDataset
from .train_model import main

if __name__ == '__main__':
    main()
