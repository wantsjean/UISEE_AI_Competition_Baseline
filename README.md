# UISEE_AI_Competition_Baseline
UISEE Self-Driving Competition Baseline

## TODO List
- [ ] Add submit testing.
- [ ] Build a faster dataset, `DALI` or `LMDB`.
- [ ] Add more data augmentation, random resize, crop, rotate, ...
- [ ] Add pretraining datasets.

# Getting Start
1. Git clone this repo.
1. Download the competition dataset from uisee.com & unzip it under `data` folder and rename it to `uisee`(for more datasets supporting)
1. Use `tiff2jpg.py` to convert and resize `tiff` to `jpg 640x320`
1. Baseline model is a simple 2D CNN + LSTM model, DIY your own model and adjust the Hyper Parameters in the `config.py` file, then start training.
