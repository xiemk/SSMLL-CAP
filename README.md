# [NeurIPS-23] Class-Distribution-Aware Pseudo-Labeling for Semi-Supervised Multi-Label Learning

The implementation for the paper [Class-Distribution-Aware Pseudo-Labeling for Semi-Supervised Multi-Label Learning](http://www.xiemk.pro/publication/neurips23-cap.pdf) (NeurIPS 2023). 

See much more related works in [Awesome Weakly Supervised Multi-Label Learning!](https://github.com/milkxie/awesome-weakly-supervised-multi-label-learning) 

## Preparing Data 

See the `README.md` file in the `data` directory for instructions on downloading and preparing the datasets.

## Training Model

To train and evaluate a model, the next two steps are required:

1. Firstly, we warm up the model with the labeled data. Run:
```
python run_warmup.py --loss_lb asl --lb_ratio 0.05 \
--warmup_epochs 12 --lr 1e-4 --net resnet50 \
--dataset_name coco --dataset_dir ./data
```

2. Secondly, we train the model with CAP method. Run:
```
python run_CAP.py --loss_lb asl --loss_ub asl --lb_ratio 0.05 \
--warmup_epochs 12 --warmup_batch_size 16 --lr 1e-4 --net resnet50 \
--dataset_name coco --dataset_dir ./data \
--init_pos_per 1.0 --init_neg_per 1.0
```

## Training Logs

### Download warmup models.

| Labeled Proportion | Warmup mAP | mAP   | Link                                                         |
| :----------------- | ---------- | ----- | ------------------------------------------------------------ |
| p=0.05             | 58.30      | 62.43 | [link](https://drive.google.com/drive/folders/1IXykq-Q6raTAU3r39R-IiKm-kJRHAUVw?usp=drive_link) |
| p=0.1              | 63.54      | 65.22 | [link](https://drive.google.com/drive/folders/1EEcIjG6qScOxIacKr2EEFm0a0CaoNpgx?usp=drive_link) |
| p=0.15             | 66.18      | 69.11 | [link](https://drive.google.com/drive/folders/10PQPWHu7RAX48v2ooih6r2u6BrxjdICh?usp=drive_link) |
| p=0.2              | 67.61      | 70.41 | [link](https://drive.google.com/drive/folders/18ZRLCmzxO4mEvdlDMdRPIKrWjfO16wUe?usp=drive_link) |



## Hyper-Parameters
To generate different entries of the main table, modify the following parameters:
1. `dataset_name`: The dataset to use, e.g. 'coco', 'voc', 'nus'.
2. `dataset_dir`: The directory of **all datasets**. 
3. `batch_size`: The batch size of samples (images).
4. `net`: The structure of model which is used to train, e.g. 'resnet50', 'mlder'(ML-Decoder with resnet50).
5. `lb_ratio`: The result of (the size of labeled samples) : (the size of all samples).
6. `loss_lb`: The loss used for labeled samples.
7. `loss_ub`: The loss used for unlabeled samples.