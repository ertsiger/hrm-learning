#!/bin/bash
python run_baseline.py --algorithm=deepsynth --domain waterworld --task='r-g' --num_instances 10 --seed 25101993 --output_dir='config/examples/baselines/04-ww-wod-rg-deepsynth' --deepsynth_cbmc_path 'cbmc/src/cbmc'
