#!/bin/sh
python3 explain_l2norm.py --dataset=synthcity --base_path=checkpoints/xai_synthcity_extract_3/ --label=both --group=both > log-synth-3-l2norm.log
python3 explain_l2norm.py --dataset=arch --base_path=checkpoints/xai_arch_extract_16/ --label=both --group=both > log-arch-16-l2norm.log
python3 explain_l2norm.py --dataset=arch --base_path=checkpoints/xai_arch_extract_17/ --label=both --group=both > log-arch-17-l2norm.log
python3 explain_l2norm.py --dataset=s3dis --base_path=checkpoints/xai_s3dis_extract_6/ --label=both --group=both > log-s3dis-6-l2norm.log