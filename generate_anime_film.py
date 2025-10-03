#!/usr/bin/env python3
"""
Wan 2.2 14B Anime Short Film Generator
High-quality video generation using A100 GPU (specifically GPU 1)
Resolution: 720p (1280x720), Anime style
"""

import os
import sys
import subprocess

# Install/upgrade required packages before importing
print("🔧 Checking and installing dependencies...")
packages = [
    "peft>=0.17.0",
    "torch>=2.0.0",
    "diffusers>=0.30.0",
    "transformers>=4.40.0",
    "accelerate>=0.30.0",
]

for package in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--quiet", package])
    except:
        print(f"⚠️  Warning: Could not install {package}")

import torch
import time
from datetime import timedelta
from pathlib import Path
import numpy as np
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from tqdm import tqdm

# Force use of GPU 1 only
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class AnimeFilmGenerator:
    def __init__(self, model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers", output_dir="./output"):
        """Initialize the generator with Wan 2.2 14B model"""
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("=" * 80)
        print("🎬 Wan 2.2 14B Anime Film Generator")
        print("=" * 80)
        print(f"📍 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"📂 Output Directory: {self.output_dir.absolute()}")
        print("=" * 80)
        
        self._load_models()
    
    def _load_models(self):
        """Load VAE, scheduler, and pipeline"""
        print("\n🔄 Loading models...")
        start_time = time.time()
        
        # Load VAE in float32 for better quality
        print("  📦 Loading VAE...")
        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_id, 
            subfolder="vae", 
            torch_dtype=torch.float32
        )
        
        # Setup scheduler for 720p (flow_shift=5.0)
        print("  ⚙️  Setting up scheduler...")
        self.scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=5.0  # 5.0 for 720P, 3.0 for 480P
        )
        
        # Load main pipeline
        print("  🚀 Loading Wan 2.2 14B pipeline...")
        self.pipe = WanPipeline.from_pretrained(
            self.model_id,
            vae=self.vae,
            torch_dtype=torch.bfloat16
        )
        self.pipe.scheduler = self.scheduler
        self.pipe.to("cuda")
        
        # Enable memory optimizations for A100
        print("  🔧 Enabling memory optimizations...")
        self.pipe.enable_model_cpu_offload()
        
        load_time = time.time() - start_time
        print(f"✅ Models loaded in {load_time:.2f}s")
        print(f"💾 Current VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    def generate_scene(self, prompt, negative_prompt=None, scene_name="scene", 
                      num_frames=81, fps=16, guidance_scale=5.0, num_inference_steps=50):
        """
        Generate a single scene from prompt
        
        Args:
            prompt: Text description of the scene (anime style)
            negative_prompt: Things to avoid in generation
            scene_name: Name for the output file
            num_frames: Number of frames (81 = ~5 seconds at 16fps)
            fps: Frames per second for output video
            guidance_scale: Classifier-free guidance scale (higher = more prompt adherence)
            num_inference_steps: Number of denoising steps (more = better quality)
        """
        if negative_prompt is None:
            # Default negative prompt for anime style
            negative_prompt = (
                "realistic photo, photorealistic, live action, 3d render, "
                "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
                "watermark, text, subtitle, logo, static, still image, "
                "jpeg artifacts, compression, oversaturated"
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
        est_time_per_step = 2.0  # seconds (conservative estimate)
        est_total = est_time_per_step * num_inference_steps
        print(f"⏱️  Estimated time: {str(timedelta(seconds=int(est_total)))}")
        
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        
        # Generate with progress tracking
        with torch.inference_mode():
            output = self.pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                height=720,
                width=1280,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).frames[0]
        
        # Calculate metrics
        generation_time = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        fps_achieved = num_frames / generation_time
        
        # Save video
        output_path = self.output_dir / f"{scene_name}.mp4"
        print(f"\n💾 Saving video to: {output_path}")
        export_to_video(output, str(output_path), fps=fps)
        
        # Print statistics
        print("\n" + "=" * 80)
        print("✅ GENERATION COMPLETE")
        print("=" * 80)
        print(f"⏱️  Generation Time: {str(timedelta(seconds=int(generation_time)))}")
        print(f"🎞️  Processing Speed: {fps_achieved:.2f} frames/second")
        print(f"💾 Peak VRAM Used: {peak_vram:.2f} GB")
        print(f"📁 Output: {output_path}")
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
