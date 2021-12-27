# Activation Modulation and Recalibration Scheme for Weakly Supervised Semantic Segmentation

This repository contains the pytorch codes and trained models described in the AAAI2021 paper ["Activation Modulation and Recalibration Scheme for Weakly Supervised Semantic Segmentation"](https://arxiv.org/abs/2112.08996)

## Preparation

#### Dependencies

```
pip install -r requirements.txt
```

#### Dataset

Download PASCAL VOC 2012 follwing instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

## Run

```
python run_sample.py --train_amr_pass True --make_cam_pass True --eval_cam_pass True
```
