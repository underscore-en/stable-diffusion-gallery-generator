import torch
import time
import random
import argparse
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from traceback import print_exc
from uuid import uuid4

"""
relevant documentations
https://huggingface.co/docs/diffusers/v0.14.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline
"""

# consts
SCHEDULER = DPMSolverMultistepScheduler
TORCH_DTYPE = torch.float16
DEFAULT_GALLERY_SIZE = 5


def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--prompt_file_path", required=True, type=str)
    parser.add_argument("--negative_prompt_file_path", required=True, type=str)
    parser.add_argument("--guidance_scale_file_path", required=True, type=str)
    parser.add_argument("--dimension_file_path", required=True, type=str)
    parser.add_argument("--num_inference_steps_file_path",
                        required=True, type=str)
    parser.add_argument("--dump_path", required=True, type=str)
    parser.add_argument("--job_file_path", required=True, type=str)
    args = parser.parse_args()

    model_path = args.model_path
    prompt_file_path = args.prompt_file_path
    negative_prompt_file_path = args.negative_prompt_file_path
    guidance_scale_file_path = args.guidance_scale_file_path
    dimension_file_path = args.dimension_file_path
    num_inference_steps_file_path = args.num_inference_steps_file_path
    dump_path = args.dump_path
    job_file_path = args.job_file_path

    # diffuser sd pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=TORCH_DTYPE,
    ).to('cuda:0')
    # use another scheduler (there is a default)
    pipeline.scheduler = SCHEDULER.from_config(pipeline.scheduler.config)
    # reduce memory usage
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_model_cpu_offload()

    # job loop
    line_ptr = 0
    while True:
        # read the line
        with open(job_file_path, "r") as f:
            lines = f.read().splitlines()
        try:
            # parse the job file
            """
            job csv file in the following format
            folder_name, count
            """
            first_line = lines[line_ptr]
            folder_name, gallery_size = first_line.split(", ")
        except Exception as e:
            print_exc()
            folder_name, gallery_size = uuid4().hex, DEFAULT_GALLERY_SIZE
        line_ptr += 1
        print(f"executing job {folder_name} ({gallery_size})")

        # clear or create the file
        gallery_folder_path = os.path.join(dump_path, folder_name)
        if os.path.exists(gallery_folder_path):
            os.unlink(gallery_folder_path)
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
        with open(guidance_scale_file_path, 'r') as f:
            guidance_scale = int(f.readline())
        with open(dimension_file_path, 'r') as f:
            dimension = tuple(int(g)
                              for g in f.read().splitlines())[0:2]
        with open(num_inference_steps_file_path, 'r') as f:
            num_inference_steps = int(f.readline())

        # image loop
        for image_count in range(gallery_size):
            try:
                # 2. contruct path
                image_path = f"{gallery_folder_path}/{image_count}.png"

                # 3. inference
                image = pipeline(
                    prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    width=dimension[0],
                    height=dimension[1],
                    num_inference_steps=num_inference_steps,
                ).images[0]

                # # save the image in 2x resolution
                # image = image.resize((dimension[0]*2, dimension[1]*2))
                image.save(image_path)

            except Exception as ex:
                # ignore
                print(ex)
                time.sleep(1)
                continue


if __name__ == "__main__":
    main()
