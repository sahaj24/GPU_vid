#!/usr/bin/env python3
"""
Wan 2.2 14B Anime Short Film Generator
High-qualit            print(f"\n📥 Downloading Wan 2.2 T2V A14B model to {self.checkpoint_dir}...") video generation using A100 GPU (specifically GPU 1)
Resolution: 720p (1280x720), Anime style

This script uses the native Wan repository instead of diffusers
"""

import os
import sys
import subprocess
import time
from datetime import timedelta
from pathlib import Path

# Force use of GPU 1 only
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print("=" * 80)
print("🎬 Wan 2.2 14B Anime Film Generator")
print("=" * 80)
print("📦 Setting up environment...")

# Clone Wan repository if not exists
wan_repo_path = Path("Wan2.1")
if not wan_repo_path.exists():
    print("📥 Cloning Wan repository...")
    subprocess.run(["git", "clone", "https://github.com/Wan-Video/Wan2.1.git"], check=True)
    os.chdir("Wan2.1")
    print("📦 Installing compatible dependencies for Python 3.8...")
    # Install compatible versions instead of using requirements.txt
    # These versions all work together without conflicts
    compatible_packages = [
        "huggingface_hub==0.20.3",  # Compatible with diffusers 0.27.2
        "peft==0.13.2",  # Keep current version
        "diffusers==0.27.2",  # Compatible version
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.5.0",
        "transformers==4.46.3",  # Latest compatible with Python 3.8
        "accelerate>=0.20.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "Pillow>=9.0.0",
        "numpy>=1.20.0",
        "imageio>=2.9.0",
        "imageio-ffmpeg>=0.4.0",
        "easydict>=1.9",  # Required by Wan
        "einops>=0.6.0",  # Required by Wan
    ]
    for pkg in compatible_packages:
        print(f"  Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg])
    os.chdir("..")
else:
    print("✅ Wan repository found")

# Add Wan to path
sys.path.insert(0, str(wan_repo_path))

import torch
import numpy as np
from tqdm import tqdm

class AnimeFilmGenerator:
    def __init__(self, checkpoint_dir="./Wan2.2-T2V-A14B", output_dir="./output"):
        """Initialize the generator with Wan 2.2 T2V A14B model (anime optimized)"""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("=" * 80)
        print("🎬 Wan 2.2 T2V A14B Anime Film Generator")
        print("=" * 80)
        print(f"📍 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"📂 Output Directory: {self.output_dir.absolute()}")
        print("=" * 80)
        
        # Download model if not exists
        if not self.checkpoint_dir.exists():
            print(f"\n� Downloading Wan 2.2 14B model to {self.checkpoint_dir}...")
            print("   This is a one-time download (~50GB)")
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"
            ])
            subprocess.run([
                "huggingface-cli", "download", "Wan-AI/Wan2.2-T2V-A14B",
                "--local-dir", str(self.checkpoint_dir)
            ], check=True)
        
        print("\n✅ Setup complete!")
    
    def generate_scene(self, prompt, negative_prompt=None, scene_name="scene", 
                      num_frames=81, fps=16, guidance_scale=5.0, num_inference_steps=50):
        """
        Generate a single scene from prompt using native Wan CLI
        
        Args:
            prompt: Text description of the scene (anime style)
            negative_prompt: Things to avoid in generation
            scene_name: Name for the output file
            num_frames: Number of frames (81 = ~5 seconds at 16fps)
            fps: Frames per second for output video
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
        """
        if negative_prompt is None:
            negative_prompt = (
                "realistic photo, photorealistic, live action, 3d render, "
                "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
                "watermark, text, subtitle, logo, static, still image"
            )
        
        # Add anime style to prompt
        full_prompt = f"Anime style, Japanese animation, {prompt}"
        
        print("\n" + "=" * 80)
        print(f"🎨 Generating Scene: {scene_name}")
        print("=" * 80)
        print(f"📝 Prompt: {full_prompt[:100]}...")
        print(f"🎞️  Frames: {num_frames} | FPS: {fps} | Duration: {num_frames/fps:.2f}s")
        print(f"📐 Resolution: 1280x720 (720p)")
        print(f"🎯 Guidance Scale: {guidance_scale} | Steps: {num_inference_steps}")
        print("=" * 80)
        
        # Calculate estimated time
        est_time_per_step = 2.0
        est_total = est_time_per_step * num_inference_steps
        print(f"⏱️  Estimated time: {str(timedelta(seconds=int(est_total)))}")
        
        start_time = time.time()
        output_path = self.output_dir / f"{scene_name}.mp4"
        
        # Run Wan generation using CLI
        cmd = [
            sys.executable,
            "Wan2.1/generate.py",
            "--task", "t2v-14B",
            "--size", "1280*720",
            "--ckpt_dir", str(self.checkpoint_dir),
            "--prompt", full_prompt,
            "--negative_prompt", negative_prompt,
            "--sample_guide_scale", str(guidance_scale),
            "--sample_steps", str(num_inference_steps),
            "--base_seed", "42",
            "--output", str(output_path)
        ]
        
        print("\n🚀 Running generation...")
        subprocess.run(cmd, check=True)
        
        generation_time = time.time() - start_time
        
        # Print statistics
        print("\n" + "=" * 80)
        print("✅ GENERATION COMPLETE")
        print("=" * 80)
        print(f"⏱️  Generation Time: {str(timedelta(seconds=int(generation_time)))}")
        print(f"📁 Output: {output_path}")
        if output_path.exists():
            print(f"📏 File Size: {output_path.stat().st_size / 1e6:.2f} MB")
        print("=" * 80)
        
        return str(output_path)
    
    def generate_film(self, scenes):
        """
        Generate multiple scenes to create a short film
        
        Args:
            scenes: List of dicts with 'name', 'prompt', and optional parameters
        """
        print("\n" + "=" * 80)
        print(f"🎬 STARTING SHORT FILM PRODUCTION")
        print(f"📊 Total Scenes: {len(scenes)}")
        print("=" * 80)
        
        film_start = time.time()
        output_files = []
        
        for i, scene in enumerate(scenes, 1):
            print(f"\n{'='*80}")
            print(f"🎬 SCENE {i}/{len(scenes)}")
            print(f"{'='*80}")
            
            scene_params = {
                'prompt': scene['prompt'],
                'scene_name': scene.get('name', f'scene_{i:02d}'),
                'negative_prompt': scene.get('negative_prompt'),
                'num_frames': scene.get('num_frames', 81),
                'fps': scene.get('fps', 16),
                'guidance_scale': scene.get('guidance_scale', 5.0),
                'num_inference_steps': scene.get('num_inference_steps', 50),
            }
            
            output_path = self.generate_scene(**scene_params)
            output_files.append(output_path)
            
            # Cleanup memory between scenes
            torch.cuda.empty_cache()
        
        total_time = time.time() - film_start
        
        print("\n" + "=" * 80)
        print("🎉 FILM PRODUCTION COMPLETE!")
        print("=" * 80)
        print(f"⏱️  Total Production Time: {str(timedelta(seconds=int(total_time)))}")
        print(f"📊 Scenes Generated: {len(output_files)}")
        print(f"📁 Output Location: {self.output_dir.absolute()}")
        print("\n📽️  Generated Videos:")
        for i, path in enumerate(output_files, 1):
            print(f"  {i}. {Path(path).name}")
        print("=" * 80)
        
        return output_files


def main():
    """Main execution function"""
    
    # Example scenes for anime short film
    # Edit the prompts below to create your custom short film
    scenes = [
        {
            'name': 'opening',
            'prompt': (
                'A peaceful Japanese village at dawn, cherry blossoms falling gently, '
                'traditional houses with curved roofs, morning mist rolling over hills, '
                'soft pastel colors, serene atmosphere, cinematic camera slowly panning'
            ),
            'num_frames': 81,
            'fps': 16,
            'num_inference_steps': 50,
        },
        {
            'name': 'character_intro',
            'prompt': (
                'Young anime girl with long flowing hair standing on a hill, '
                'wind blowing her hair and school uniform, determined expression, '
                'holding a katana, dramatic lighting from behind, '
                'vibrant colors, dynamic pose, studio ghibli inspired'
            ),
            'num_frames': 81,
            'fps': 16,
            'num_inference_steps': 50,
        },
        {
            'name': 'action_scene',
            'prompt': (
                'Epic anime battle scene, magical energy particles flying, '
                'dynamic camera movement, speed lines, intense action, '
                'bright colorful effects, dramatic shadows and highlights, '
                'fluid animation style, high-energy combat'
            ),
            'num_frames': 81,
            'fps': 16,
            'num_inference_steps': 50,
        },
        {
            'name': 'emotional_moment',
            'prompt': (
                'Two anime characters facing each other under moonlight, '
                'emotional scene, tears glistening, soft bokeh background, '
                'warm color grading, intimate moment, gentle wind, '
                'studio quality animation, detailed facial expressions'
            ),
            'num_frames': 81,
            'fps': 16,
            'num_inference_steps': 50,
        },
        {
            'name': 'ending',
            'prompt': (
                'Sunset over the ocean, anime characters silhouette walking together, '
                'orange and pink sky, peaceful ending, hopeful atmosphere, '
                'credits-worthy scene, beautiful composition, '
                'wide cinematic shot, perfect lighting'
            ),
            'num_frames': 81,
            'fps': 16,
            'num_inference_steps': 50,
        },
    ]
    
    # Initialize generator
    generator = AnimeFilmGenerator()
    
    # Generate the film
    output_files = generator.generate_film(scenes)
    
    print("\n" + "=" * 80)
    print("🎊 ALL DONE! Your anime short film has been generated!")
    print("=" * 80)
    print("\n💡 TIP: You can combine these scenes using ffmpeg:")
    print(f"   cd {generator.output_dir}")
    print("   ffmpeg -f concat -safe 0 -i <(printf \"file '%s'\\n\" *.mp4) -c copy final_film.mp4")
    print("=" * 80)


if __name__ == "__main__":
    main()
