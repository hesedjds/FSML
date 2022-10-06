# Few-shot metric learning

This repository is the official implementation of Few-shot Metric Learning

To install requirements:

```setup
conda env create -f environment.yml
```

## Install dataset 
Download the CUB_200_2011.tgz file from http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz 
and place it to filelists/CUB folder.

```install 
cd filelists/CUB
bash download_CUB.sh
cd ../miniImagenet
bash download_miniImagenet.sh
```

## Training

To train the model(s) in the paper, run this command:

```train
python train_metric.py --dataset <CUB/miniImagenet> --method <sft/crml> --train_n_way 5 --test_n_way 5 --n_shot 5 --train_aug --batch_size 300 
```
if you train the CRML, then add --pretrained save/<mini/cub>_sft.tar to load the pretrained model. 

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python test_all.py --dataset miniImagenet --method sft --train_n_way 5 --test_n_way 5 --n_shot 5 --train_aug --path save/mini_sft.tar --iter_num 600 
python test_all.py --dataset miniImagenet --method crml --train_n_way 5 --test_n_way 5 --n_shot 5 --train_aug --path save/mini_crml_5w5s.tar --crml_lr 0.025 --iter_num 600 
python test_all.py --dataset CUB --method sft --train_n_way 5 --test_n_way 5 --n_shot 5 --train_aug --path save/mini_sft.tar --iter_num 600 
python test_all.py --dataset CUB --method crml --train_n_way 5 --test_n_way 5 --n_shot 5 --train_aug --path save/cub_crml_5w5s.tar --crml_lr 0.025 --iter_num 600 

```

If you run the above script then you can check the performance in the main paper. 

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 