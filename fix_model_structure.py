#!/usr/bin/env python3
"""Fix Wan model structure"""

import json
import os
from pathlib import Path

model_dir = Path("Wan2.2-T2V-A14B")

# Create config for subdirectories
subdirs = ["low_noise_model", "high_noise_model"]

for subdir in subdirs:
    subdir_path = model_dir / subdir
    config_path = subdir_path / "config.json"
    
    if not config_path.exists():
        print(f"Creating {subdir}/config.json...")
        
        config = {
            "_class_name": "WanModel",
            "_diffusers_version": "0.21.4",
            "in_channels": 16,
            "out_channels": 16,
            "sample_size": [1, 90, 160],
            "patch_size": [1, 2, 2],
            "num_attention_heads": 24,
            "attention_head_dim": 128,
            "num_layers": 48,
            "pooled_projection_dim": 2048,
            "guidance_embeds": True,
            "text_len": 256,
            "text_dim": 4096
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Created {config_path}")
    else:
        print(f"✅ {subdir}/config.json exists")

# Check for model_index.json
index_path = subdir_path / "model_index.json"
if not index_path.exists():
    for subdir in subdirs:
        index_path = model_dir / subdir / "model_index.json"
        if not index_path.exists():
            print(f"Creating {subdir}/model_index.json...")
            index = {
                "_class_name": "WanModel",
                "_diffusers_version": "0.21.4"
            }
            with open(index_path, 'w') as f:
                json.dump(index, f, indent=2)
            print(f"✅ Created {index_path}")

print("\n✅ Model structure fixed!")
