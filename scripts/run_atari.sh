#!/usr/bin/env bash

# Atari experiments (Default: Use 20 expert demo)

# Set working directory to iq_learn
# cd ..

# Pong
uv run train_iq.py agent=softq env=pong agent.init_temp=1e-3 method.loss=value_expert method.chi=True seed=0

# Breakout
uv run train_iq.py agent=softq env=breakout agent.init_temp=1e-3 method.loss=value_expert method.chi=True seed=0

# Space Invaders
uv run train_iq.py agent=softq env=space agent.init_temp=1e-3 method.loss=value_expert method.chi=True seed=0
