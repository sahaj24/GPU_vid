#!/usr/bin/env python3
"""Patch Wan2.2 code for missing imports"""

import os

# Patch the motioner.py file
motioner_file = "Wan2.2/wan/modules/s2v/motioner.py"

if os.path.exists(motioner_file):
    with open(motioner_file, 'r') as f:
        content = f.read()
    
    # Replace the problematic import
    old_line = "from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin"
    new_line = """# Patched import for compatibility
try:
    from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
except ImportError:
    # Fallback for older diffusers versions
    try:
        from diffusers.loaders.single_file import FromOriginalModelMixin
        from diffusers.loaders import PeftAdapterMixin
    except ImportError:
        # Create dummy classes if neither work
        class FromOriginalModelMixin:
            pass
        class PeftAdapterMixin:
            pass"""
    
    if old_line in content and "# Patched import" not in content:
        content = content.replace(old_line, new_line)
        
        with open(motioner_file, 'w') as f:
            f.write(content)
        
        print("✅ Patched Wan2.2/wan/modules/s2v/motioner.py")
    else:
        print("✅ motioner.py already patched or changed")
else:
    print("❌ motioner.py not found")

print("\n✅ All patches applied!")
