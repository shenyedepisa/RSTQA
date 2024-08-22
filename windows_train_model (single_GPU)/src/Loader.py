import os
import json
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import torch
from .answerEncoder import answerNumber


class Loader(Dataset):
    def __init__(
            self,
            config,
            DataConfig,
            seqEncoder,
            img_size,
            textHead,
            imageHead,
            train=True,
            transform=None,
    ):
        self.img_size = img_size
        self.Encoder = seqEncoder
        self.textHead = textHead
        self.imgHead = imageHead
        self.imgFolder = config["new_data_path"]
        self.questions_file = DataConfig["questionsJSON"]
        self.images_file = DataConfig["imagesJSON"]
        self.imageFile = config["DataConfig"]["images_path"]
        self.imgSource = config["DataConfig"]["sourceMask_path"]
        self.imgTarget = config["DataConfig"]["targetMask_path"]
        self.imgBackground = config["DataConfig"]["backgroundMask_path"]
        self.train = train
        self.transform = transform
        self.addMask = config["add_mask"]
        self.answerEncoder = answerNumber(config, config["DataConfig"]["answersJson"])

        with open(self.questions_file) as json_data:
            self.questionsJSON = json.load(json_data)
        with open(self.images_file) as json_data:
            self.imagesJSON = json.load(json_data)

        self.imageActive = [img["id"] for img in self.imagesJSON["images"] if img["active"]]
        self.questionActive = [q["id"] for q in self.questionsJSON["questions"] if q["active"]]
        self.length = len(self.questionActive)
        self.questions = self.questionsJSON["questions"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        question = self.questions[idx]

        image = question['img_id']
        img = Image.open(os.path.join(self.imageFile, str(image) + ".png"))
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img = np.array(img)
        target = np.array(Image.open(os.path.join(self.imgTarget, str(image) + ".png")))
        source = np.array(Image.open(os.path.join(self.imgSource, str(image) + ".png")))
        background = np.array(Image.open(os.path.join(self.imgBackground, str(image) + ".png")))

        source = source[:, :, np.newaxis]
        target = target[:, :, np.newaxis]
        background = background[:, :, np.newaxis]

        source_mask = source + background * 0.1
        target_mask = target + background * 0.1
        if self.addMask:
            background_mask = background * 0.1 + source * 0.7 + target * 0.9
        else:
            background_mask = background * 0.1

        # mask = np.concatenate((source_mask, target_mask, background_mask), axis=-1).astype(np.uint8)
        # The mask has not been normalized.
        # This may be modified in future versions, but currently this method works better than directly normalizing the mask
        mask = np.concatenate((source_mask, target_mask, background_mask), axis=-1)

        sourceImage = T.ToTensor()(img)
        mask = self.transform["mask"](mask).float()
        imgT = self.transform["image"](img.copy())

        Question = self.Encoder.encode(question['question'], question=True)
        if self.textHead == "siglip-512":
            Question["input_ids"] = (
                torch.as_tensor(np.array(Question["input_ids"])).long().squeeze(0)
            )
        else:
            Question["input_ids"] = (
                torch.as_tensor(np.array(Question["input_ids"])).long().squeeze(0)
            )
            Question["attention_mask"] = (
                torch.as_tensor(np.array(Question["attention_mask"])).long().squeeze(0)
            )
        answer = self.answerEncoder.encode(question['type'], question['answer'])
        answer = torch.as_tensor(np.array(answer)).long()
        if self.train:
            return (Question, answer, imgT, mask)
        else:
            return (Question, answer, imgT, question["type"], mask, sourceImage)
