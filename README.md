## RFM Generation Script
This repository aims to provide a minimal script to extract region features out of images (or frames) using the latest object detection methods. It also provides a minimal use case of the **PyTables** library for storing the region features on hard disk. That's because the region features, especially when working with video can become very memory intensive sometimes. Note that the object detection code heavily borrows from Detectron 2 framework by Facebook.

## Installation
1. Clone the repository:

```
git clone https://github.com/aligholami/RFMGen.git && cd RFMGen/
```

2. Install the requirements:

```
pip install -r requirements.txt
```

#### Requirements
This repository is tested with the latest `Python 3.7.5` version. Following are the requirements for this script:
```
tables==3.6.1
detectron2.egg==info
torch==1.3.1
tqdm==4.40.2
torchvision==0.4.2
```

## Usage
To use the script, you are free to change every part of it. By default, there are some command line arguments provided for quick usage:
```
python main.py --dataset DATA_SET_NAME --dataset-path PATH_TO_SAVE_DOWNLOADED_DATASET --batch-size BATCH_SIZE --lr LEARNING_RATE --rpn-config-path PATH_TO_DETECTRON_MODEL_CONFIG --rpn-pretrained-path PATH_TO_PRETRAINED_DETECTRON_MODEL --num-max-regions MAX_NUM_REGIONS_PER_IMAGE --rf-path PATH_TO_REGIONS_DIRECTORY --no-cuda USE_CUDA_OR_NOT (True or False)
``` 
