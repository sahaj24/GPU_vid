import os

# Fix the diffusers import in Wan2.2's code
file_path = "Wan2.2/wan/modules/model.py"

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the problematic import line
    old_import = "from diffusers.configuration_utils import ConfigMixin, register_to_config"
    new_import = """# Patched for compatibility
try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
except:
    from diffusers import ConfigMixin
    def register_to_config(cls):
        return cls"""
    
    if old_import in content and "# Patched for compatibility" not in content:
        content = content.replace(old_import, new_import)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✅ Patched Wan2.2/wan/modules/model.py")
    else:
        print("✅ Already patched or file has changed")
else:
    print("❌ File not found")

# Also fix the diffusers dynamic_modules_utils.py file
diffusers_file = "/home/srmist17/.local/lib/python3.8/site-packages/diffusers/utils/dynamic_modules_utils.py"

if os.path.exists(diffusers_file):
    with open(diffusers_file, 'r') as f:
        content = f.read()
    
    # Replace cached_download with hf_hub_download
    if "from huggingface_hub import cached_download" in content:
        content = content.replace(
            "from huggingface_hub import cached_download, hf_hub_download, model_info",
            "from huggingface_hub import hf_hub_download, model_info\ncached_download = hf_hub_download  # Compatibility fix"
        )
        
        with open(diffusers_file, 'w') as f:
            f.write(content)
        
        print("✅ Patched diffusers/utils/dynamic_modules_utils.py")
    else:
        print("✅ diffusers already patched")
else:
    print("⚠️ diffusers file not found at expected location")
