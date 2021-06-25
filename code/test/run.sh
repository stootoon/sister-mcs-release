#!/bin/bash
rm -rf params
rm -f *.npy
rm -f *.pdf
python -u ../run_sisters.py params.json --write_every 100
