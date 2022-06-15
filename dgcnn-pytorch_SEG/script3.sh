#!/bin/sh
python explain.py --dataset=synthcity --base_path=checkpoints/xai_synthcity_extract_3/ --tipo=umap --label=pred --use_pca=True --layers=5 > log-synth-3-all-l5.log
python explain.py --dataset=arch --base_path=checkpoints/xai_arch_extract_16/ --tipo=both --label=both --use_pca=True --layers=5 > log-arch-16-all-l5.log
python explain.py --dataset=arch --base_path=checkpoints/xai_arch_extract_17/ --tipo=both --label=both --use_pca=True --layers=5 > log-arch-17-all-l5.log
python explain.py --dataset=s3dis --base_path=checkpoints/xai_s3dis_extract_6/ --tipo=both --label=both --use_pca=True --layers=5 > log-s3dis-6-all-l5.log