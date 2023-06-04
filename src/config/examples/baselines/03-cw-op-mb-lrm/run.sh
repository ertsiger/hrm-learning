#!/bin/bash
python run_baseline.py --algorithm=lrm --domain craftworld --task='milk-bucket' --num_instances 10 --seed 25101993 --output_dir='config/examples/03-cw-op-mb-lrm' --lrm_max_states 4
