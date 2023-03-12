import torch
import time
import datetime
import os
import json
import argparse
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

"""
documentation for the pipeline
https://huggingface.co/docs/diffusers/v0.14.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline
"""

# consts
SCHEDULER = DPMSolverMultistepScheduler
TORCH_DTYPE = torch.float16

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--prompt_file_path", required=True, type=str)
    parser.add_argument("--negative_prompt_file_path", required=True, type=str)
    parser.add_argument("--guidance_scale_file_path", required=True, type=str)
    parser.add_argument("--dimension_file_path", required=True, type=str)
    parser.add_argument("--num_inference_steps_file_path", required=True, type=str)
    parser.add_argument("--dump_path", required=True, type=str)
    args = parser.parse_args()
    model_path = args.model_path
    prompt_file_path = args.prompt_file_path
    negative_prompt_file_path = args.negative_prompt_file_path
    guidance_scale_file_path = args.guidance_scale_file_path
    dimension_file_path = args.dimension_file_path
    num_inference_steps_file_path = args.num_inference_steps_file_path
    dump_path = args.dump_path

    # diffuser sd pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=TORCH_DTYPE,
    ).to('cuda:0')
    pipeline.scheduler = SCHEDULER.from_config(pipeline.scheduler.config)
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_model_cpu_offload()

    prompt = None
    negative_prompt = None
    guidance_scale = None
    dimension = None
    num_inference_steps = None

    prefix = None
    counter = 0

    while True:
        try:
            # 1. if anything argument new, use new parameter
            new_prompt = None
            new_negative_prompt = None
            new_guidance_scale = None
            new_dimension = None
            new_num_inference_steps = None
            with open(prompt_file_path, 'r') as f:
                new_prompt = ", ".join(f.read().splitlines())
            with open(negative_prompt_file_path, 'r') as f:
                new_negative_prompt = ", ".join(f.read().splitlines())
            with open(guidance_scale_file_path, 'r') as f:
                new_guidance_scale = int(f.readline())
            with open(dimension_file_path, 'r') as f:
                new_dimension = tuple(int(g) for g in f.read().splitlines())[0:2]
            with open(num_inference_steps_file_path, 'r') as f:
                new_num_inference_steps = int(f.readline())
            if new_prompt != prompt or new_negative_prompt != negative_prompt or guidance_scale != new_guidance_scale or dimension != new_dimension or num_inference_steps != new_num_inference_steps:
                prompt, negative_prompt, guidance_scale, dimension, num_inference_steps = new_prompt, new_negative_prompt, new_guidance_scale, new_dimension, new_num_inference_steps
                prefix = datetime.datetime.now().strftime("h_%Y%m%d_%H%M%S")
                counter = 0

            # 2. contruct path
            counter += 1
            path = f"{dump_path}/{prefix}_{counter}.png"
            print(path)

            # 3. inference
            image = pipeline(
                prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                width=dimension[0],
                height=dimension[1],
                num_inference_steps=num_inference_steps,
            ).images[0]

            # save the image in 2x resolution
            image.save(path)

        except Exception as ex:
            # ignore
            print(ex)
            time.sleep(1)
            continue
