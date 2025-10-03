#!/usr/bin/env python3
"""
Simple Wan 2.2 14B Video Generation Script
Uses diffusers library directly - proven working approach
Single file, GPU 1 only
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Force GPU 1 usage
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

def install_dependencies():
    """Install required packages if not present"""
    print("📦 Checking dependencies...")
    
    packages = [
        "torch",
        "diffusers>=0.30.0",
        "transformers",
        "accelerate",
        "opencv-python",
        "imageio",
        "imageio-ffmpeg",
        "Pillow",
        "numpy",
    ]
    
    # Suppress warnings during import check
    import warnings
    warnings.filterwarnings('ignore')
    
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import diffusers
    except ImportError:
        missing.append("diffusers")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import accelerate
    except ImportError:
        missing.append("accelerate")
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import imageio
    except ImportError:
        missing.append("imageio")
    
    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("Installing required packages...")
        for pkg in packages:
            subprocess.run([sys.executable, "-m", "pip", "install", "-U", pkg], 
                         check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("\n⚠️  Please restart the script after dependencies are installed")
        sys.exit(0)
    
    print("✅ All dependencies installed")

def generate_video(
    prompt,
    output_path="output.mp4",
    num_frames=81,
    height=720,
    width=1280,
    fps=16,
    guidance_scale=5.0,
    num_inference_steps=50,
    seed=42
):
    """
    Generate video using Wan 2.2 14B model via diffusers
    """
    print("\n" + "="*80)
    print(f"📝 Prompt: {prompt}")
    print(f"🎞️  Frames: {num_frames} | FPS: {fps} | Duration: {num_frames/fps:.2f}s")
    print(f"📐 Resolution: {width}x{height}")
    print(f"🎯 Guidance Scale: {guidance_scale} | Steps: {num_inference_steps}")
    print("="*80)
    
    try:
        import torch
        from diffusers import DiffusionPipeline
        from diffusers.utils import export_to_video
        
        print(f"\n🔧 Loading model on GPU 1...")
        print(f"💾 VRAM before: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
        
        # Load the pipeline from local directory
        model_path = "./Wan2.2-T2V-A14B"
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            print(f"Please download the model first or check the path")
            return None
        
        print(f"📂 Loading from local path: {model_path}")
        
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        
        print(f"✅ Model loaded successfully")
        print(f"💾 VRAM after loading: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
        
        # Set seed for reproducibility
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Generate video
        print(f"\n🚀 Generating video...")
        start_time = time.time()
        
        # Add negative prompt for better quality
        negative_prompt = (
            "realistic photo, photorealistic, live action, 3d render, "
            "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
            "watermark, text, subtitle, logo"
        )
        
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        frames = output.frames[0]
        
        generation_time = time.time() - start_time
        
        print(f"✅ Generation complete in {generation_time:.1f}s")
        print(f"💾 Peak VRAM: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
        
        # Save video
        print(f"\n💾 Saving video to: {output_path}")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        export_to_video(frames, output_path, fps=fps)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Video saved: {output_path} ({file_size:.2f} MB)")
        
        # Clean up
        del pipe
        del output
        torch.cuda.empty_cache()
        
        return output_path
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    print("🎬 Wan 2.2 14B Anime Video Generator")
    print("=" * 80)
    
    # Install dependencies
    install_dependencies()
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Define anime scenes
    scenes = [
        {
            "name": "opening",
            "prompt": (
                "Anime style, Japanese animation, A peaceful Japanese village at dawn, "
                "cherry blossoms falling gently, traditional houses with curved roofs, "
                "morning mist rolling over hills, soft pastel colors, serene atmosphere, "
                "cinematic camera slowly panning"
            ),
            "num_frames": 81,
            "fps": 16,
        },
        {
            "name": "character_intro",
            "prompt": (
                "Anime style, Japanese animation, Young anime girl with long black hair, "
                "school uniform, standing on a bridge, wind blowing through hair, "
                "sakura petals swirling around, soft lighting, beautiful detailed eyes, "
                "gentle smile, cinematic composition"
            ),
            "num_frames": 81,
            "fps": 16,
        },
        {
            "name": "action_scene",
            "prompt": (
                "Anime style, Japanese animation, Dynamic action sequence, anime character "
                "running through city streets, speed lines, motion blur effect, "
                "vibrant colors, dramatic lighting, energetic movement, "
                "urban background, fast-paced animation"
            ),
            "num_frames": 81,
            "fps": 16,
        },
    ]
    
    print(f"\n📊 Total scenes to generate: {len(scenes)}")
    total_duration = sum(s['num_frames'] / s['fps'] for s in scenes)
    print(f"⏱️  Total video duration: {total_duration:.1f}s")
    
    # Generate each scene
    start_time = time.time()
    generated_files = []
    
    for i, scene in enumerate(scenes, 1):
        print(f"\n{'='*80}")
        print(f"🎬 Scene {i}/{len(scenes)}: {scene['name']}")
        print(f"{'='*80}")
        
        output_path = output_dir / f"{scene['name']}.mp4"
        
        result = generate_video(
            prompt=scene['prompt'],
            output_path=str(output_path),
            num_frames=scene['num_frames'],
            height=720,
            width=1280,
            fps=scene['fps'],
            guidance_scale=5.0,
            num_inference_steps=50,
            seed=42 + i
        )
        
        if result:
            generated_files.append(result)
        else:
            print(f"⚠️  Failed to generate scene: {scene['name']}")
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*80)
    print("🎉 GENERATION COMPLETE!")
    print("="*80)
    print(f"✅ Generated {len(generated_files)}/{len(scenes)} scenes")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📂 Output directory: {output_dir.absolute()}")
    print("\n📹 Generated files:")
    for file in generated_files:
        size = os.path.getsize(file) / (1024 * 1024)
        print(f"  • {file} ({size:.2f} MB)")
    print("="*80)

if __name__ == "__main__":
    main()
