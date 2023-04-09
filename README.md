# stable-diffusion-gallery-generator

Generate a variety of galleries with stable diffusion.

### Main Script `./main.py`

This script runs the gallery generation logic.

```python
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--prompt_file_path", required=True, type=str)
    parser.add_argument("--negative_prompt_file_path", required=True, type=str)
    parser.add_argument("--gallery_dump_path", required=True, type=str)
    parser.add_argument("--job_file_path", required=True, type=str)
    parser.add_argument("--upscale", action="store_true")
```

```bash
py ./main.py --model_path "C:\Users\Private\Desktop\stable_diffusion\models\basemodel\p" --prompt_file_path ".\config_private\prompt.txt" --negative_prompt_file_path ".\config_private\negative_prompt.txt" --gallery_dump_path "C:\Users\Private\Desktop\fh\gallery" --job_file_path "./config_private/jobs.txt"
py ./main.py --model_path "C:\Users\Private\Desktop\stable_diffusion\models\composed\a_ct2" --prompt_file_path ".\config_private\prompt.txt" --negative_prompt_file_path ".\config_private\negative_prompt.txt" --gallery_dump_path "C:\Users\Private\Desktop\fh\gallery" --job_file_path "./config_private/jobs.txt"
py ./main.py --model_path "C:\Users\Private\Desktop\stable_diffusion\models\composed\p_ip" --prompt_file_path ".\config_private\prompt.txt" --negative_prompt_file_path ".\config_private\negative_prompt.txt" --gallery_dump_path "C:\Users\Private\Desktop\fh\gallery" --job_file_path "./config_private/jobs.txt"
```

#### Utility Script './scripts/convert_original_stable_diffusion_to_diffusers.py`

This script is copied from diffuser repository as a utility to convert safetensor files from civitai to diffuser format.

```bash
py .\scripts\convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "C:\Users\Private\Desktop\stable_diffusion\models\basemodel\p.safetensors" --dump_path "C:\Users\Private\Desktop\stable_diffusion\models\basemodel\p" --from_safetensors
```

#### Lora + Base Model Composition `./scripts/convert_lora_safetensor_to_diffuser.py`

```bash
py .\scripts\convert_lora_safetensor_to_diffuser.py --base_model_path "C:\Users\Private\Desktop\stable_diffusion\models\basemodel\a" --checkpoint_path "C:\Users\Private\Desktop\stable_diffusion\models\loras\ct_lora.safetensors" --alpha 0.2 --dump_path "C:\Users\Private\Desktop\stable_diffusion\models\composed\a_ct2"
py .\scripts\convert_lora_safetensor_to_diffuser.py --base_model_path "C:\Users\Private\Desktop\stable_diffusion\models\basemodel\p" --checkpoint_path "C:\Users\Private\Desktop\stable_diffusion\models\loras\ip_lora.safetensors" --alpha 0.7 --dump_path "C:\Users\Private\Desktop\stable_diffusion\models\composed\p_ip"
```

## TODO: Terminologies

### vae

