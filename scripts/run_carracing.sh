#!/usr/bin/env bash

# Atari experiments (Default: Use 20 expert demo)

# Set working directory to iq_learn
cd ..

# CARRACING
python train_iq.py env=carracing agent=sac expert.demos=20 method.loss=v0 method.regularize=True agent.actor_lr=3e-05 seed=0


