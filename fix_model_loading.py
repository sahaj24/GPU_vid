#!/usr/bin/env python3
"""Fix Wan2.2 model loading to use subdirectories"""

import os

# Check the text2video.py to see how it loads models
text2video_file = "Wan2.2/wan/text2video.py"

if os.path.exists(text2video_file):
    with open(text2video_file, 'r') as f:
        content = f.read()
    
    # Find the model loading line
    old_line1 = 'self.low_noise_model = WanModel.from_pretrained('
    old_line2 = 'self.high_noise_model = WanModel.from_pretrained('
    
    # Check if it needs patching
    if 'low_noise_model' in content and '# Patched model loading' not in content:
        # Patch to load from subdirectories
        content = content.replace(
            'self.low_noise_model = WanModel.from_pretrained(\n            os.path.join(checkpoint_dir, "low_noise_model")',
            'self.low_noise_model = WanModel.from_pretrained(\n            os.path.join(checkpoint_dir, "low_noise_model")  # Patched model loading'
        )
        
        content = content.replace(
            'self.high_noise_model = WanModel.from_pretrained(\n            os.path.join(checkpoint_dir, "high_noise_model")',
            'self.high_noise_model = WanModel.from_pretrained(\n            os.path.join(checkpoint_dir, "high_noise_model")  # Patched model loading'
        )
        
        # If it's loading from root, change to subdirectory
        content = content.replace(
            'self.low_noise_model = WanModel.from_pretrained(\n            checkpoint_dir',
            'self.low_noise_model = WanModel.from_pretrained(\n            os.path.join(checkpoint_dir, "low_noise_model")  # Patched to use subdirectory'
        )
        
        content = content.replace(
            'self.high_noise_model = WanModel.from_pretrained(\n            checkpoint_dir',
            'self.high_noise_model = WanModel.from_pretrained(\n            os.path.join(checkpoint_dir, "high_noise_model")  # Patched to use subdirectory'
        )
        
        with open(text2video_file, 'w') as f:
            f.write(content)
        
        print("✅ Patched text2video.py model loading")
    else:
        print("✅ Already patched or code structure changed")
        
    # Show what the code looks like now
    print("\nChecking model loading code...")
    with open(text2video_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'low_noise_model = WanModel.from_pretrained' in line or 'high_noise_model = WanModel.from_pretrained' in line:
                print(f"Line {i+1}: {line.rstrip()}")
                if i+1 < len(lines):
                    print(f"Line {i+2}: {lines[i+1].rstrip()}")
else:
    print("❌ text2video.py not found")
