#!/bin/sh
python explain_pca.py --dataset=synthcity --base_path=checkpoints/xai_synthcity_extract_3/ --label=pred --layers=1,2,3,5,4 --type_class=-1 > log-synth-3-pca.log
python explain_pca.py --dataset=arch --base_path=checkpoints/xai_arch_extract_16/ --label=both --layers=1,2,3,5,4 --type_class=-1 > log-arch-16-pca.log
python explain_pca.py --dataset=arch --base_path=checkpoints/xai_arch_extract_17/ --label=both --layers=1,2,3,5,4 --type_class=-1 > log-arch-17-pca.log
python explain_pca.py --dataset=s3dis --base_path=checkpoints/xai_s3dis_extract_6/ --label=both --layers=1,2,3,5,4 --type_class=-1 > log-s3dis-6-pca.log 