import os
import sys

# Check if Wan2.1 directory exists locally
wan_path = "./Wan2.1"
if os.path.exists(wan_path):
    print(f"✅ Found Wan2.1 directory")
    
    # Look for important files
    files_to_check = [
        "README.md",
        "generate.py", 
        "requirements.txt",
        "wan/modules/model.py"
    ]
    
    for file in files_to_check:
        full_path = os.path.join(wan_path, file)
        if os.path.exists(full_path):
            print(f"\n{'='*60}")
            print(f"📄 {file}")
            print('='*60)
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if len(content) > 2000:
                    print(content[:2000] + f"\n... (truncated, total {len(content)} chars)")
                else:
                    print(content)
else:
    print(f"❌ Wan2.1 directory not found at {wan_path}")
