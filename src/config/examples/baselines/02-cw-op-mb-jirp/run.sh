#!/bin/bash
python run_baseline.py --algorithm=jirp --domain craftworld --task='milk-bucket' --num_instances 10 --seed 25101993 --output_dir='config/examples/baselines/02-cw-op-mb-jirp' --jirp_max_states 4
