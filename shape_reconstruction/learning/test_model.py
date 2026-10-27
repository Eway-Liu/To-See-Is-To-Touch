"""Compatibility entry point for shared evaluation/inference."""
from .inference import ReconstructionModel
from .evaluation import main

if __name__ == '__main__':
    main()
