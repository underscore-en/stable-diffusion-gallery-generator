import torch
import time
import argparse
import os
import random
import itertools
from math import floor
from diffusers import DPMSolverMultistepScheduler, TextToVideoZeroPipeline, StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline, EulerAncestralDiscreteScheduler
from typing import Tuple
from uuid import uuid4
from lpw_pipeline import StableDiffusionLongPromptWeightingPipeline

"""
relevant documentations
https://huggingface.co/docs/diffusers/v0.14.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline
"""

"""
helper functions
"""
def toValidDimension(dimension: Tuple[int, int]) -> Tuple[int, int]:
    """
    multiple of 8 per spec
    """
    return tuple(8*floor(d/8) for d in dimension)

def inference_steps_of(d):
    return floor((30 * (d[0] + d[1]) / 1600 )**0.9)

"""
consts
"""
OVERNIGHT_BATCH_SIZE = 5
DIMENSION_GENERATOR = lambda: random.choice([(800, 800), (1200, 800), (800, 1200)])
SCHEDULER = EulerAncestralDiscreteScheduler
# SCHEDULER = DPMSolverMultistepScheduler
TORCH_DTYPE = torch.float16
UPSCALE_FACTOR = 2

def DYNAMIC_PARSE_STRATEGY(lines):
    guidance_scale = int(lines[0])
    prompt = ", ".join(lines[1:])
    return guidance_scale, prompt

def OVERNIGHT_PARSE_STRATEGY(lines):
    guidance_scale = int(lines[0])
    prompt_lines = list()
    buffer = list()
    use_buffer = False
    for line in lines[1:]:
        if line == "@":
            if not use_buffer:
                # flush
                prompt_lines.append(random.choice(buffer))

            use_buffer = not use_buffer
        if use_buffer:
            buffer.append(line)
        else:
            prompt_lines.append(line)

    return guidance_scale, ", ".join(prompt_lines)

def generate_batch(pipeline_t2i, dimension, prompt: str, negative_prompt: str, guidance_scale, gallery_dump_path: str, batch_size: int, output_prefix: str):
    num_inference_steps = inference_steps_of(dimension)

    # job loop
    for count in range(batch_size):
        try:
            # 2. contruct path
            image_path = os.path.join(gallery_dump_path, f"{output_prefix}_{count}.png")
            print(image_path)

            """
            3. inference
            """
            image = pipeline_t2i(
                prompt,
                width=dimension[0],
                height=dimension[1],
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_embeddings_multiples=100,
            ).images[0]

            image = image.resize(toValidDimension(tuple(UPSCALE_FACTOR * d for d in dimension)))
            image.save(image_path)

        except Exception as ex:
            # ignore
            print(ex)
            time.sleep(1)
            continue
    

def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--overnight", action="store_true")
    parser.add_argument("--prompt_file_path", required=True, type=str)
    parser.add_argument("--negative_prompt_file_path", required=True, type=str)
    parser.add_argument("--gallery_dump_path", required=True, type=str)
    args = parser.parse_args()

    model_dir = args.model_dir
    model_name = args.model_name
    overnight = args.overnight
    prompt_file_path = args.prompt_file_path
    negative_prompt_file_path = args.negative_prompt_file_path
    gallery_dump_path = args.gallery_dump_path

    if not overnight:
        if not model_name:
            raise Exception("model name required")
        
        # regular execution model

        # t2i pipeline
        pipeline_t2i: StableDiffusionLongPromptWeightingPipeline = StableDiffusionLongPromptWeightingPipeline.from_pretrained(
            os.path.join(model_dir, model_name),
            torch_dtype=TORCH_DTYPE,
        ).to('cuda')
        pipeline_t2i.scheduler = SCHEDULER.from_config(
            pipeline_t2i.scheduler.config)
        dimension = DIMENSION_GENERATOR()
        prefix = f"{now}_{model_name}"
 
        while True:
            with open(prompt_file_path, 'r') as f:
                lines = f.read().splitlines()
            guidance_scale, prompt = DYNAMIC_PARSE_STRATEGY(lines)
            with open(negative_prompt_file_path, 'r') as f:
                negative_prompt = ", ".join(f.read().splitlines())
            generate_batch(pipeline_t2i, dimension, prompt, negative_prompt, guidance_scale, gallery_dump_path, 1, prefix)
    else:
        # overnight execution model
        while (True):
            dimension = DIMENSION_GENERATOR()
            now = time.time()
            for modelname in os.listdir(model_dir):
                # get model
                model_path = os.path.join(model_dir, modelname)
                if not os.path.isdir(model_path):
                    print(f"{model_path} is not a model");
                    continue
                prefix = f"{now}_{modelname}"

                pipeline_t2i: StableDiffusionLongPromptWeightingPipeline = StableDiffusionLongPromptWeightingPipeline.from_pretrained(
                    model_path,
                    torch_dtype=TORCH_DTYPE,
                ).to('cuda')
                pipeline_t2i.scheduler = SCHEDULER.from_config(
                    pipeline_t2i.scheduler.config)

                with open(prompt_file_path, 'r') as f:
                    lines = f.read().splitlines()
                guidance_scale, prompt = OVERNIGHT_PARSE_STRATEGY(lines)
                with open(negative_prompt_file_path, 'r') as f:
                    negative_prompt = ", ".join(f.read().splitlines())

                generate_batch(pipeline_t2i, dimension, prompt, negative_prompt, guidance_scale, gallery_dump_path, OVERNIGHT_BATCH_SIZE, prefix)


    


if __name__ == "__main__":
    main()
