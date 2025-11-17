# Commands for Model Variants

## Single-View Model
### Train
python3 train.py --ymlpath=./experiment/singleview2500/d2_singleview2500.yml --gpu=0 --dataroot=./data/LIDC-HDF5-256 --dataset=train --tag=d2_singleview2500 --data=LIDC256 --dataset_class=align_ct_xray_std --model_class=SingleViewCTGAN --datasetfile=./data/train.txt --valid_datasetfile=./data/test.txt --valid_dataset=test

### Test
python3 test.py --ymlpath=./experiment/singleview2500/d2_singleview2500.yml --gpu=0 --dataroot=./data/LIDC-HDF5-256 --dataset=test --tag=d2_singleview2500 --data=LIDC256 --dataset_class=align_ct_xray_std --model_class=SingleViewCTGAN --datasetfile=./data/test.txt --resultdir=./singleview --check_point=30 --how_many=3

## Multi-View Model
### Train
python3 train.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml --gpu=0 --dataroot=./data/LIDC-HDF5-256 --dataset=train --tag=d2_multiview2500 --data=LIDC256 --dataset_class=align_ct_xray_views_std --model_class=MultiViewCTGAN --datasetfile=./data/train.txt --valid_datasetfile=./data/test.txt --valid_dataset=test

### Test
python3 test.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml --gpu=0 --dataroot=./data/LIDC-HDF5-256 --dataset=test --tag=d2_multiview2500 --data=LIDC256 --dataset_class=align_ct_xray_views_std --model_class=MultiViewCTGAN --datasetfile=./data/test.txt --resultdir=./multiview --check_point=90 --how_many=3

## GAN-dataset-complete Single-View Model
### Train
python3 train.py --ymlpath=./experiment/singleview2500/d2_singleview2500.yml --gpu=0 --dataroot=./data/GAN-dataset-complete --dataset=train --tag=singleview_gan_dataset --data=GAN-dataset-complete --dataset_class=align_ct_xray_std --model_class=SingleViewCTGAN --datasetfile=./data/synthetic_train.txt --valid_datasetfile=./data/synthetic_test.txt --valid_dataset=test

### Test
python3 test.py --ymlpath=./experiment/singleview2500/d2_singleview2500.yml --gpu=0 --dataroot=./data/GAN-dataset-complete --dataset=test --tag=singleview_gan_dataset --data=GAN-dataset-complete --dataset_class=align_ct_xray_std --model_class=SingleViewCTGAN --datasetfile=./data/synthetic_test.txt --resultdir=./singleview_gan_dataset --check_point=30 --how_many=3

## GAN-dataset-complete Multi-View Model
### Train
python3 train.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml --gpu=0 --dataroot=./data/GAN-dataset-complete --dataset=train --tag=multiview_gan_dataset --data=GAN-dataset-complete --dataset_class=align_ct_xray_views_std --model_class=MultiViewCTGAN --datasetfile=./data/synthetic_train.txt --valid_datasetfile=./data/synthetic_test.txt --valid_dataset=test

### Test
python3 test.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml --gpu=0 --dataroot=./data/GAN-dataset-complete --dataset=test --tag=multiview_gan_dataset --data=GAN-dataset-complete --dataset_class=align_ct_xray_views_std --model_class=MultiViewCTGAN --datasetfile=./data/synthetic_test.txt --resultdir=./multiview_gan_dataset --check_point=90 --how_many=3

## GAN-dataset-complete-real Single-View Model
### Train
python3 train.py --ymlpath=./experiment/singleview2500/d2_singleview2500.yml --gpu=0 --dataroot=./data/GAN-dataset-complete-real --dataset=train --tag=singleview_gan_dataset_real --data=GAN-dataset-complete-real --dataset_class=align_ct_xray_std --model_class=SingleViewCTGAN --datasetfile=./data/real_train.txt --valid_datasetfile=./data/real_test.txt --valid_dataset=test

### Test
python3 test.py --ymlpath=./experiment/singleview2500/d2_singleview2500.yml --gpu=0 --dataroot=./data/GAN-dataset-complete-real --dataset=test --tag=singleview_gan_dataset_real --data=GAN-dataset-complete-real --dataset_class=align_ct_xray_std --model_class=SingleViewCTGAN --datasetfile=./data/real_test.txt --resultdir=./singleview_gan_dataset_real --check_point=30 --how_many=3

## GAN-dataset-complete-real Multi-View Model
### Train
python3 train.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml --gpu=0 --dataroot=./data/GAN-dataset-complete-real --dataset=train --tag=multiview_gan_dataset_real --data=GAN-dataset-complete-real --dataset_class=align_ct_xray_views_std --model_class=MultiViewCTGAN --datasetfile=./data/real_train.txt --valid_datasetfile=./data/real_test.txt --valid_dataset=test

### Test
python3 test.py --ymlpath=./experiment/multiview2500/d2_multiview2500.yml --gpu=0 --dataroot=./data/GAN-dataset-complete-real --dataset=test --tag=multiview_gan_dataset_real --data=GAN-dataset-complete-real --dataset_class=align_ct_xray_views_std --model_class=MultiViewCTGAN --datasetfile=./data/real_test.txt --resultdir=./multiview_gan_dataset_real --check_point=90 --how_many=3

