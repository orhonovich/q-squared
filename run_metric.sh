#!/bin/bash

python pipeline/run_pipeline.py \
      --infile data/dodeca_inconsistent.csv \
      --gen_method beam \
      --q_per_cand single \
      --personal remove \
      --outfile dodeca_dev_no_hallu_no_coref_beam_single_remove.csv