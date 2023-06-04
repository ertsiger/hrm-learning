#!/bin/bash
python run_baseline.py --algorithm=deepsynth --domain craftworld --task='milk-bucket' --num_instances 10 --seed 25101993 --output_dir='config/examples/baselines/01-cw-op-mb-deepsynth' --deepsynth_cbmc_path 'cbmc/src/cbmc'
