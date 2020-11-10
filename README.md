![image](https://github.com/chencheng1203/DeLF-pytorch/blob/master/results/RANSAC_match.png)

### Image Retrival with DeLF
The pytorch code for image retrival with [DeLF](https://arxiv.org/abs/1612.06321), this repository can be as a good baseline for the learning of image retrival. The training stages were splited into single file, so you can get a clear idea of the work flow, the code sometimes tedious but simple.some code were used from [https://github.com/nashory/DeLF-pytorch](https://github.com/nashory/DeLF-pytorch)

### requirements
- pytorch >= 1.3 & torchvision
- opencv-python
- fassi
- scipy
- scikit-learn
- skimage
- h5py


### note
For the convenience of debugging, argparse is not used, config.py file instead,for different stage, there are 4 config.py file uesd:
- train_cfg.py -> for training stage, fineturn with Google Landmark dataset and train attention layers
- extract_cfg.py -> for pca training and delf feature extraction
- match_cfg.py -> build retrival system config

### training
In the stage of finetune and keypoint attention, you should change the config in train_cfg.py. 
1. fineturn
```
cd train
python train_ft.py
```
2. train keypoint model
```
cd train
python train_atte.py
```

### extract
```
cd extract and 'change extract_cfg.py first'
```
1. train pca
```
python train_pca.py
```
2. extract delf feature
```
python store_delf_feature.py
```

### image retrival and keypoints match
For inference details, A jupyter-notebook was prepared, you can go to <kbd>scripts/match.ipynb</kbd> for detailed matching process. In the file <kbd>scripts/delf_feature_visulization.ipynb</kdb> delf keypoints feature also has shown.
```
cd match
python match.py
```

### DeLF keypoints Visualize
![image](https://github.com/chencheng1203/DeLF-pytorch/blob/master/results/attention_visualize.png)

### Image Retrival Results
![image](https://github.com/chencheng1203/DeLF-pytorch/blob/master/results/retrival_results.png)
note: because only 2 guangzhou tower index images were saved, but get top 3 retrival resluts, the last one obvious wrong.
