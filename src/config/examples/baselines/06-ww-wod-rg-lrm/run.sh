#!/bin/bash
python run_baseline.py --algorithm=lrm --domain waterworld --task='r-g' --num_instances 10 --seed 25101993 --output_dir='config/examples/baselines/06-ww-wod-rg-lrm' --lrm_max_states 3
