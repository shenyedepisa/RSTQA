import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import copy
import torchvision.transforms as T
from src import Loader, SeqEncoder, seed_torch, ex
from train import train
from train_qa import train2

seed_torch(3407)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    # RGB
    IMAGENET_MEAN = [0.3833698, 0.39640951, 0.36896593]
    IMAGENET_STD = [0.21045856, 0.1946447, 0.18824594]

    _config['cd'] = os.getcwd()
    image_size = _config["image_resize"]
    source_img_size = _config["source_image_size"]
    textHead = _config["textHead"]
    imageHead = _config["imageHead"]
    Data = _config["DataConfig"]

    print(f"textModel: {textHead}")
    print(f"imageModel: {imageHead}")
    data_transforms = {
        "image": T.Compose(
            [
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                T.Resize((image_size, image_size), antialias=True),
            ]
        ),
        "mask": T.Compose(
            [T.ToTensor(), T.Resize((image_size, image_size), antialias=True)]
        ),
    }
    seq_Encoder = SeqEncoder(_config, Data["allQuestionsJSON"], textTokenizer=textHead)
    print("Training dataset preprocessing...")
    train_dataset = Loader(
        _config,
        Data["train"],
        seq_Encoder,
        source_img_size,
        textHead=textHead,
        imageHead=imageHead,
        train=True,
        transform=data_transforms,
    )
    print("Validation dataset preprocessing...")
    val_dataset = Loader(
        _config,
        Data["val"],
        seq_Encoder,
        source_img_size,
        textHead=textHead,
        imageHead=imageHead,
        train=False,
        transform=data_transforms,
    )
    print("Testing dataset preprocessing...")
    test_dataset = Loader(
        _config,
        Data["test"],
        seq_Encoder,
        source_img_size,
        textHead=textHead,
        imageHead=imageHead,
        train=False,
        transform=data_transforms,
    )

    train(_config, train_dataset, val_dataset, test_dataset, seq_Encoder)
    oneStep = _config["one_step"]
    if not oneStep:
        time.sleep(50)
        train2(_config, train_dataset, val_dataset, test_dataset, seq_Encoder)
