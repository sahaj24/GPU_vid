#!/usr/bin/env python3
"""Check and fix Wan model structure"""

import os
import json
from pathlib import Path

model_dir = Path("Wan2.2-T2V-A14B")

print(f"📂 Checking model directory: {model_dir}")

if not model_dir.exists():
    print("❌ Model directory not found")
    exit(1)

# List all files
print("\n📋 Files in model directory:")
files = sorted(model_dir.rglob("*"))
for f in files[:20]:  # Show first 20
    if f.is_file():
        size = f.stat().st_size / (1024**2)
        print(f"  {f.relative_to(model_dir)} ({size:.1f} MB)")

# Check for config.json
config_path = model_dir / "config.json"
if not config_path.exists():
    print("\n⚠️  config.json not found - creating it...")
    
    # Create a basic config for Wan model
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
    
    print(f"✅ Created config.json")
else:
    print("\n✅ config.json exists")
    with open(config_path) as f:
        config = json.load(f)
    print(f"Config: {json.dumps(config, indent=2)}")
