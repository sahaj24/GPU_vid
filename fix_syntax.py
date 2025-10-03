#!/usr/bin/env python3
"""Fix the syntax error in text2video.py"""

import os
import re

text2video_file = "Wan2.2/wan/text2video.py"

if os.path.exists(text2video_file):
    with open(text2video_file, 'r') as f:
        content = f.read()
    
    # Fix the broken lines - find the problematic pattern and fix it
    # The issue is the comment was inserted in the middle of a function call
    
    # Pattern: WanModel.from_pretrained(\n            os.path.join(...) # comment, subfolder=...
    # Should be: WanModel.from_pretrained(\n            os.path.join(...), subfolder=...
    
    # Fix low_noise_model
    content = re.sub(
        r'(self\.low_noise_model = WanModel\.from_pretrained\(\s+os\.path\.join\(checkpoint_dir, "low_noise_model"\))\s*#[^\n]*,\s*subfolder=',
        r'\1,\n            subfolder=',
        content
    )
    
    # Fix high_noise_model
    content = re.sub(
        r'(self\.high_noise_model = WanModel\.from_pretrained\(\s+os\.path\.join\(checkpoint_dir, "high_noise_model"\))\s*#[^\n]*,\s*subfolder=',
        r'\1,\n            subfolder=',
        content
    )
    
    with open(text2video_file, 'w') as f:
        f.write(content)
    
    print("✅ Fixed syntax error in text2video.py")
    
    # Verify the fix
    print("\nVerifying fix...")
    with open(text2video_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[95:115], start=96):
            print(f"Line {i}: {line.rstrip()}")
else:
    print("❌ text2video.py not found")
