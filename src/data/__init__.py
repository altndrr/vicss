from src.data.ade20k_150 import ADE20K150
from src.data.caltech101 import Caltech101
from src.data.dtd import DTD
from src.data.eurosat import EuroSAT
from src.data.fgvc_aircraft import FGVCAircraft
from src.data.flowers102 import Flowers102
from src.data.food101 import Food101
from src.data.imagenet import ImageNet
from src.data.oxford_pets import OxfordPets
from src.data.pascal_context_59 import PASCALContext59
from src.data.pascal_voc_20 import PascalVOC20
from src.data.stanford_cars import StanfordCars
from src.data.sun397 import SUN397
from src.data.ucf101 import UCF101

__all__ = [
    "ADE20K150",
    "Caltech101",
    "DTD",
    "EuroSAT",
    "FGVCAircraft",
    "Food101",
    "Flowers102",
    "ImageNet",
    "OxfordPets",
    "PASCALContext59",
    "PascalVOC20",
    "StanfordCars",
    "SUN397",
    "UCF101",
]

DATA = {
    "ade20k_150": ADE20K150,
    "caltech101": Caltech101,
    "dtd": DTD,
    "eurosat": EuroSAT,
    "fgvc_aircraft": FGVCAircraft,
    "food101": Food101,
    "flowers102": Flowers102,
    "imagenet": ImageNet,
    "oxford_pets": OxfordPets,
    "pascal_context_59": PASCALContext59,
    "pascal_voc_20": PascalVOC20,
    "stanford_cars": StanfordCars,
    "sun397": SUN397,
    "ucf101": UCF101,
}
