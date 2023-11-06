# EPIC Fields and Neural Rendering

## About

This repository contains code for model training and inference with [EPIC Fields](https://arxiv.org/abs/2306.08731).

## Setup

Use the following commands to setup the folder structure for:
- model checkpoints
- data split
- COLMAP (EPIC Fields) reconstructions

Make sure to set `DIR_EK100` as it is linked in the commands. This directory contains the [EPIC-KITCHENS-100 dataset](https://data.bris.ac.uk/data/dataset/2g1n6qdydwa9u22shpxqzp0t8m). It should contain tar files and be in the form `EpicKitchens-100/{pid}/rgb_frames/{vid}.tar` where pid and vid represent the Person ID and Video ID respectively.

If you don't have a google account or haven't installed gdrive for downloading files from google drive via CLI, then download the files manually with the provided links (they are publicly accessible).

```
# download and extract the model checkpoints
# https://drive.google.com/file/d/1O_SygdKstj2J7QG89yYabnib2G6rbjhU/view?usp=sharing
gdrive download 1O_SygdKstj2J7QG89yYabnib2G6rbjhU
tar -xvf ckpts.tar

# setup data directory
mkdir data
cd data

# split
# https://drive.google.com/file/d/1aUF2M3VHqYP4_zebVdMYNWaikPHRI_XY/view?usp=sharing
gdrive download 1aUF2M3VHqYP4_zebVdMYNWaikPHRI_XY
tar -xzvf split.tar.gz

# link EpicKitchens-100
ln -s $DIR_EK100 ek100

# link COLMAP reconstructions (containing the 50 reconstructions: P01_01, P01_09, etc.)
ln -s $COLMAP_RECONSTRUCTIONS reconstructions
```

## Training and Inference

The notebook `experiment` in `notebooks` provides examples for:
- training a model
- continuing training a model from a checkpoint
- inferencing with a trained model (loaded from the provided checkpoints)
