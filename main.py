import torch
import time
import random
import argparse
import os
from math import floor
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionUpscalePipeline
from traceback import print_exc
from uuid import uuid4
from lpw_pipeline import StableDiffusionLongPromptWeightingPipeline

"""
relevant documentations
https://huggingface.co/docs/diffusers/v0.14.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline
"""

# consts
SCHEDULER = DPMSolverMultistepScheduler
TORCH_DTYPE = torch.float16
DEFAULT_GALLERY_SIZE = 20

DIMENSION = (512, 512)
GUIDANCE_SCALE = 4
NUM_INFERENCE_STEPS = 35
UPSCALE_FACTOR = 2
I2I_STRENGTH = 0.7


def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--prompt_file_path", required=True, type=str)
    parser.add_argument("--negative_prompt_file_path", required=True, type=str)
    parser.add_argument("--gallery_dump_path", required=True, type=str)
    parser.add_argument("--job_file_path", required=True, type=str)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    model_path = args.model_path
    prompt_file_path = args.prompt_file_path
    negative_prompt_file_path = args.negative_prompt_file_path
    gallery_dump_path = args.gallery_dump_path
    job_file_path = args.job_file_path
    fast = args.fast

    # t2i pipeline
    pipeline_t2i = StableDiffusionLongPromptWeightingPipeline.from_pretrained(
        model_path,
        torch_dtype=TORCH_DTYPE,
    ).to('cuda')
    pipeline_t2i.scheduler = SCHEDULER.from_config(
        pipeline_t2i.scheduler.config)
    pipeline_t2i.enable_xformers_memory_efficient_attention()

    # i2i pipeline
    pipeline_i2i = None
    if not fast:
        pipeline_i2i: StableDiffusionImg2ImgPipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=TORCH_DTYPE
        ).to("cuda:0")
        pipeline_i2i.scheduler = SCHEDULER.from_config(
            pipeline_t2i.scheduler.config)
        pipeline_i2i.enable_xformers_memory_efficient_attention()
        pipeline_i2i.enable_model_cpu_offload()

    # job loop
    job_ptr = 0
    while True:
        # read the line
        with open(job_file_path, "r") as f:
            csv_lines = f.read().splitlines()
        try:
            # parse the job file
            """
            job csv file in the following format
            folder_name, count
            """
            first_line = csv_lines[job_ptr]
            folder_name, _gallery_size = first_line.split(", ")
            gallery_size = int(_gallery_size)
        except Exception as e:
            print_exc()
            folder_name, gallery_size = uuid4().hex, DEFAULT_GALLERY_SIZE
        job_ptr += 1
        print(f"executing job {folder_name} ({gallery_size})")

        # clear or create the file
        gallery_folder_path = os.path.join(gallery_dump_path, folder_name)
        if os.path.exists(gallery_folder_path):
            for file in os.listdir(gallery_folder_path):
                os.remove(os.path.join(gallery_folder_path, file))
        else:
            os.mkdir(gallery_folder_path)

        # inference arguments
        with open(prompt_file_path, 'r') as f:
            prompts = []
            for line in f.read().splitlines():
                choices = line.split(", ")
                choice = random.choice(choices)
                if choice.strip():
                    prompts.append(choice)
                print(len(prompts), choice, choices)
            prompt = ", ".join(prompts)
        with open(negative_prompt_file_path, 'r') as f:
            negative_prompt = ", ".join(f.read().splitlines())

        # image loop
        for image_count in range(gallery_size):
            try:
                # 2. contruct path
                image_path = f"{gallery_folder_path}/{image_count+1}.png"
                print(image_path)

                """
                3. inference
                From experience, to generate directly with 1024*1024 will have too much details.
                """
                image = pipeline_t2i(
                    prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                ).images[0]

                """
                4. upscale
                https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img

                """

                image = image.resize(
                    (
                        8 * floor(DIMENSION[0]*UPSCALE_FACTOR/8),
                        8 * floor(DIMENSION[1]*UPSCALE_FACTOR/8),
                    )
                )

                if not fast:
                    image = pipeline_i2i(
                        prompt,
                        image=image,
                        negative_prompt=negative_prompt,
                        guidance_scale=GUIDANCE_SCALE,
                    ).images[0]

                image.save(image_path)

            except Exception as ex:
                # ignore
                print(ex)
                time.sleep(1)
                continue


if __name__ == "__main__":
    main()
