# [NeurIPS-23] Class-Distribution-Aware Pseudo Labeling for Semi-Supervised Multi-Label Learning

The implementation for the paper [Class-Distribution-Aware Pseudo Labeling for Semi-Supervised Multi-Label Learning](https://arxiv.org/pdf/2305.02795.pdf) (NeurIPS 2023). 

See much more related works in [Awesome Weakly Supervised Multi-Label Learning!](https://github.com/milkxie/awesome-weakly-supervised-multi-label-learning) 

## Preparing Data 

See the `README.md` file in the `data` directory for instructions on downloading and preparing the datasets.

## Training Model

To train and evaluate a model, the next two steps are required:

1. Fristly, we warm-up the model with the labeled data. Run:
```
python run_warmup.py
```

2. Then, we train the model using CAP method. Run:
```
python run_cap.py
```

<!-- ## Hyper-Parameters
To obtain the results reported in the paper, please modify the following parameters:
1. `dataset_name`: The dataset to use, e.g. 'coco', 'voc', 'nus', 'cub'.
2. `dataset_dir`: The directory of **all datasets**. 
3. `batch_size`: The batch size of samples (images).
3. `lambda_plc`: The weight of PLC regularization item.
4. `lambda_lac`: The weight of LAC regularization item.
4. `threshold`: The threshold for pseudo positive labels.
4. `temperature`: The temperature for LAC regularization.
4. `queue_size`: The size of the Memory Queue.
4. `is_proj`: The switch of the projector which generates label-wise embeddings.
4. `is_data_parallel`: The switch of training with multi-GPUs. -->


<!-- ## Misc

* The range of hyper-parameters can be found in the paper.
* There are four folders in this directory `dataset_dir` --- 'coco/', 'voc/', 'nus/', 'cub/'. Please make sure that the path of the dataset is correct before training.
* We performed all experiments on two GeForce RTX 3090 GPUs, so the `os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"`. The switch of training with multi-GPUs is `False` by default, and you can open it with `--is_data_parallel`. -->

<!-- ## Reference
If you find the code useful in your research, please consider citing our paper:
```
@inproceedings{
	xie2022labelaware,
	title={Label-Aware Global Consistency for Multi-Label Learning with Single Positive Labels},
	author={Ming-Kun Xie and Jia-Hao Xiao and Sheng-Jun Huang},
	booktitle={Advances in Neural Information Processing Systems},
	year={2022}
}
``` -->