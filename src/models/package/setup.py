from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ["wandb==0.15.11", "tensorflow-hub==0.14.0"]

setup(
    name="multimodal_fakenews_detector_model_training",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="Multimodal_Fakenews_Detector Trainer Application",
)
