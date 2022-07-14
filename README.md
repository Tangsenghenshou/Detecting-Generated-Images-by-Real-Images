# Detecting-Generated-Images-by-Real-Images
## Setup
python >= 3.8
torch >= 1.8.0
## Make LNP dataset
We have extracted the LNP using the code in [CycleISP](https://github.com/swz30/CycleISP), replacing the corresponding code in CycleISP with our provided `dataset_rgb.py`, `denoising_rgb.py` and `test_sidd_rgb.py`.  
```python
python test_sidd_rgb.py
```  
## Classification
Take the produced LNP dataset and sort it using the code provided by [CNNDetection](https://github.com/peterwang512/CNNDetection). Use `resnet.py` to replace the code in it.  
```python
python train.py --name model --dataroot ./dataset/ --classes airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse
 ```
 ## Acknowledgments
 This repository borrows content from [CycleISP](https://github.com/swz30/CycleISP), as well as the [CNNDetection](https://github.com/peterwang512/CNNDetection) repository.
