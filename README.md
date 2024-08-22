# Global-TQA: Global Remote Sensing Image Tampering Question and Answering

This is the initial version of the Global-TQA dataset, Global-TQA-B dataset and Tampering Awareness Framework (TQF). 

### Installation

##### python >=3.10

```
conda create -n tamper python=3.10
conda activate tamper
```

##### pytorch

[**install pytorch**](https://pytorch.org/)

```
# e.g. CUDA 11.8
# with conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# with pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

##### Install Packages

```
pip install -r requirements.txt
```

### Download Datasets

- **Datasets V1.0 is released at Baidu Drive**

  [512×512 (77.6G)](https://pan.baidu.com/s/1O_SSvcjyUYVnn65iRxQpIA?pwd=AAAI)

  [224×224 (15.6G)](https://pan.baidu.com/s/1W9T7i5sKo97i-COfktWrAg?pwd=AAAI)

- **TBD**: Datasets V2.0 will be released at Google Drive and Baidu Drive before December 19, 2024.

- Dataset Directory: ` datasets/`

- Dataset Subdirectory: `datasets/JsonFiles/`, `datasets/JsonFilesBalanced/`, `datasets/image/`, `datasets/source/`, `datasets/target/`, `datasets/background/`


### Download pre-trained weights

[**Download clip-b-32 weights from Hugging Face**](https://huggingface.co/openai/clip-vit-base-patch32/tree/main)

- Clip Directory: `models/clipModels/openai_clip_b_32/`

[**Download U-Net weights from Github**](https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth) 

- U-Net Directory: `models/imageModels/milesial_UNet/`

### Start Training

```
python main.py
```

- Modify the experiment settings and hyperparameters in `src/config.py`

### Data Examples

<img src="https://ice.frostsky.com/2024/08/22/994e02007d47c474045cdd598afb0982.png" alt="appendix" height="500" />



### Citation

```
@unpublished{global2025aaai,
  title = {Global-TQA: Global Remote Sensing Image Tampering Question and Answering},
  url={https://anonymous.4open.science/r/TQA},
  month = {8},
  year = {2024}
}
```

### License

[**CC BY-NC-SA 4.0**](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)

All images and their associated annotations in Global-TQA can be used for academic purposes only, **but any commercial use is prohibited.**
