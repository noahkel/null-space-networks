#!/usr/bin/env python3
"""Generate the single-rectangle phantom dataset.

Thin wrapper around the shared pipeline in create_phantom_data.py;
accepts the same command-line arguments.
"""
from create_phantom_data import main

if __name__ == "__main__":
    main(shape="rectangles")
