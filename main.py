import torch
import time
import argparse
import os
from math import floor
from diffusers import DPMSolverMultistepScheduler, StableDiffusionUpscalePipeline
from typing import Tuple
from uuid import uuid4
from lpw_pipeline import StableDiffusionLongPromptWeightingPipeline

"""
relevant documentations
https://huggingface.co/docs/diffusers/v0.14.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline
"""


def toValidDimension(dimension: Tuple[int, int]) -> Tuple[int, int]:
    """
    returns the dimension tuple in valid format
    validity is defined by the acceptable dimension in diffuser api, which are multiples of 8.
    """
    return tuple(8*floor(d/8) for d in dimension)


# consts
SCHEDULER = DPMSolverMultistepScheduler
TORCH_DTYPE = torch.float16

DIMENSION = toValidDimension((512, 512))
# DIMENSION = toValidDimension((640, 640))
NUM_INFERENCE_STEPS = 25
UPSCALE_FACTOR = 2
I2I_STRENGTH = 0.7


def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--prompt_file_path", required=True, type=str)
    parser.add_argument("--negative_prompt_file_path", required=True, type=str)
    parser.add_argument("--gallery_dump_path", required=True, type=str)
    parser.add_argument("--upscale", action="store_true")
    args = parser.parse_args()

    model_path = args.model_path
    prompt_file_path = args.prompt_file_path
    negative_prompt_file_path = args.negative_prompt_file_path
    gallery_dump_path = args.gallery_dump_path
    upscale = args.upscale

    # t2i pipeline
    pipeline_t2i: StableDiffusionLongPromptWeightingPipeline = StableDiffusionLongPromptWeightingPipeline.from_pretrained(
        model_path,
        torch_dtype=TORCH_DTYPE,

    ).to('cuda')
    pipeline_t2i.scheduler = SCHEDULER.from_config(
        pipeline_t2i.scheduler.config)
    pipeline_t2i.enable_xformers_memory_efficient_attention()

    # upscale pipeline
    pipeline_us = None
    if upscale:
        pipeline_us = StableDiffusionUpscalePipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", revision="fp16", torch_dtype=torch.float16
        ).to('cuda')
        pipeline_us.enable_attention_slicing()
        pipeline_us.enable_xformers_memory_efficient_attention()
        pipeline_us.enable_sequential_cpu_offload()

    # job loop
    folder_name = None
    guidance_scale = None
    prompt = None
    counter = 0
    while True:
        with open(prompt_file_path, 'r') as f:
            _gs, *l = f.read().splitlines()
            gs = int(_gs)
            p = ", ".join(l)
            if p != prompt or guidance_scale != gs:
                folder_name = uuid4().hex
                guidance_scale = gs
                prompt = p
                counter = 0
                gallery_folder_path = os.path.join(
                    gallery_dump_path, folder_name)
                os.mkdir(gallery_folder_path)
                print(prompt)

        with open(negative_prompt_file_path, 'r') as f:
            negative_prompt = ", ".join(f.read().splitlines())

        # image loop
        try:
            # 2. contruct path
            image_path = f"{gallery_folder_path}/{counter+1}.png"
            print(image_path)

            """
            3. inference
            From experience, to generate directly with 1024*1024 will have too much details.
            """
            image = pipeline_t2i(
                prompt,
                width=DIMENSION[0],
                height=DIMENSION[1],
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=NUM_INFERENCE_STEPS,
                max_embeddings_multiples=200
            ).images[0]

            # image = image.resize(toValidDimension(
            #     tuple(UPSCALE_FACTOR * d for d in DIMENSION))
            # )
            image.save(image_path)

            """
            4. upscale
            """
            if upscale:
                image = pipeline_us(
                    prompt="",
                    negative_prompt=negative_prompt,
                    image=image,
                    guidance_scale=guidance_scale
                ).images[0]
                image.save(image_path)

            counter += 1

        except Exception as ex:
            # ignore
            print(ex)
            time.sleep(1)
            continue


if __name__ == "__main__":
    main()
