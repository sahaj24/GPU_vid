#!/usr/bin/env python3
"""
Wan 2.2 14B Video Generator - With Fixed Dependencies
Patches Wan's code to work with modern huggingface_hub
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Force GPU 1 usage
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def patch_wan_code():
    """Patch Wan's code to use hf_hub_download instead of cached_download"""
    print("🔧 Patching Wan code for modern huggingface_hub...")
    
    file_path = "Wan2.1/wan/modules/model.py"
    
    if not os.path.exists(file_path):
        print("❌ Wan code not found")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the import
    if 'from diffusers.configuration_utils import ConfigMixin, register_to_config' in content:
        # Check if already patched
        if 'from huggingface_hub import hf_hub_download' in content:
            print("✅ Already patched")
            return True
        
        # Replace cached_download with hf_hub_download
        content = content.replace(
            'from diffusers.configuration_utils import ConfigMixin, register_to_config',
            '''from diffusers.configuration_utils import ConfigMixin, register_to_config
# Patched for modern huggingface_hub compatibility
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)'''
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✅ Patched successfully")
        return True
    
    return False

def check_wan_repo():
    """Check if Wan repository exists, clone if not"""
    if not os.path.exists("Wan2.1"):
        print("📥 Cloning Wan2.1 repository...")
        subprocess.run(["git", "clone", "https://github.com/wapchief/Wan.git", "Wan2.1"], check=True)
        print("✅ Repository cloned")
    else:
        print("✅ Wan2.1 repository found")

def check_model():
    """Check if model exists"""
    if not os.path.exists("Wan2.2-T2V-A14B"):
        print("❌ Model not found at ./Wan2.2-T2V-A14B")
        print("Please download the model first")
        sys.exit(1)
    print("✅ Model found: Wan2.2-T2V-A14B")

def install_deps():
    """Install any modern versions - we'll patch the code"""
    print("\n📦 Installing dependencies...")
    
    packages = [
        "diffusers",
        "transformers",
        "accelerate", 
        "torch",
        "easydict",
        "einops",
    ]
    
    for pkg in packages:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    print("✅ Dependencies installed")

def generate_video(
    prompt,
    output_path,
    negative_prompt="",
    size="1280*720",
    num_frames=81,
    fps=16,
    guidance_scale=5.0,
    steps=50,
    seed=42
):
    """Generate video using Wan's generate.py"""
    
    print("\n" + "="*80)
    print(f"📝 Prompt: {prompt[:80]}...")
    print(f"🎞️  Frames: {num_frames} | FPS: {fps} | Duration: {num_frames/fps:.2f}s")
    print(f"📐 Resolution: {size.replace('*', 'x')}")
    print(f"🎯 Guidance Scale: {guidance_scale} | Steps: {steps}")
    print("="*80)
    
    # Build command with Python path to ensure correct environment
    cmd = [
        sys.executable,
        "Wan2.1/generate.py",
        "--task", "t2v-14B",
        "--size", size,
        "--ckpt_dir", "Wan2.2-T2V-A14B",
        "--prompt", prompt,
        "--negative_prompt", negative_prompt,
        "--sample_guide_scale", str(guidance_scale),
        "--sample_steps", str(steps),
        "--base_seed", str(seed),
        "--output", output_path,
    ]
    
    print(f"\n🚀 Generating video...")
    start_time = time.time()
    
    try:
        # Run with output displayed
        result = subprocess.run(cmd, check=True)
        
        generation_time = time.time() - start_time
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\n✅ Generation complete in {generation_time:.1f}s")
            print(f"💾 Video saved: {output_path} ({file_size:.2f} MB)")
            return True
        else:
            print(f"\n❌ Output file not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Generation failed with exit code {e.returncode}")
        return False

def main():
    print("🎬 Wan 2.2 14B Anime Video Generator (Fixed)")
    print("=" * 80)
    
    # Setup
    check_wan_repo()
    check_model()
    patch_wan_code()
    install_deps()
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Define scenes
    scenes = [
        {
            "name": "opening",
            "prompt": (
                "Anime style, Japanese animation, A peaceful Japanese village at dawn, "
                "cherry blossoms falling gently, traditional houses with curved roofs, "
                "morning mist rolling over hills, soft pastel colors, serene atmosphere, "
                "cinematic camera slowly panning"
            ),
        },
        {
            "name": "character_intro",
            "prompt": (
                "Anime style, Japanese animation, Young anime girl with long black hair, "
                "school uniform, standing on a bridge, wind blowing through hair, "
                "sakura petals swirling around, soft lighting, beautiful detailed eyes, "
                "gentle smile, cinematic composition"
            ),
        },
        {
            "name": "action_scene",
            "prompt": (
                "Anime style, Japanese animation, Dynamic action sequence, anime character "
                "running through city streets, speed lines, motion blur effect, "
                "vibrant colors, dramatic lighting, energetic movement, "
                "urban background, fast-paced animation"
            ),
        },
    ]
    
    negative_prompt = (
        "realistic photo, photorealistic, live action, 3d render, "
        "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
        "watermark, text, subtitle, logo"
    )
    
    print(f"\n📊 Total scenes to generate: {len(scenes)}")
    
    # Generate
    start_time = time.time()
    success_count = 0
    
    for i, scene in enumerate(scenes, 1):
        print(f"\n{'='*80}")
        print(f"🎬 Scene {i}/{len(scenes)}: {scene['name']}")
        print(f"{'='*80}")
        
        output_path = str(output_dir / f"{scene['name']}.mp4")
        
        if generate_video(
            prompt=scene['prompt'],
            output_path=output_path,
            negative_prompt=negative_prompt,
            size="1280*720",
            num_frames=81,
            fps=16,
            guidance_scale=5.0,
            steps=50,
            seed=42 + i
        ):
            success_count += 1
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*80)
    print("🎉 GENERATION COMPLETE!")
    print("="*80)
    print(f"✅ Generated {success_count}/{len(scenes)} scenes")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📂 Output directory: {output_dir.absolute()}")
    
    if success_count > 0:
        print("\n📹 Generated files:")
        for scene in scenes:
            file_path = output_dir / f"{scene['name']}.mp4"
            if file_path.exists():
                size = file_path.stat().st_size / (1024 * 1024)
                print(f"  • {file_path} ({size:.2f} MB)")
    print("="*80)

if __name__ == "__main__":
    main()
