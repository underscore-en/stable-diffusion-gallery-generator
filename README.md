# stable-diffusion-gallery-generator

Generate a variety of galleries with stable diffusion.

### Main Script `./main.py`

This script runs the gallery generation logic.

```python
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--overnight", action="store_true")
    parser.add_argument("--prompt_file_path", required=True, type=str)
    parser.add_argument("--negative_prompt_file_path", required=True, type=str)
    parser.add_argument("--gallery_dump_path", required=True, type=str)
```

```bash
py ./main.py --model_dir "C:\Users\Private\Desktop\stable_diffusion\models" --model_name "h"  --prompt_file_path ".\config_private\prompt.txt" --negative_prompt_file_path ".\config_private\negative_prompt.txt" --gallery_dump_path "C:\Users\Private\Desktop\stable_diffusion\gallery"

py ./main.py --model_dir "C:\Users\Private\Desktop\stable_diffusion\models" --overnight --prompt_file_path ".\config_private\prompt_overnight.txt" --negative_prompt_file_path ".\config_private\negative_prompt.txt" --gallery_dump_path "C:\Users\Private\Desktop\stable_diffusion\gallery"
```

#### Utility Script './scripts/convert_original_stable_diffusion_to_diffusers.py`

This script is copied from diffuser repository as a utility to convert safetensor files from civitai to diffuser format.

```bash
py .\scripts\convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "C:\Users\Private\Desktop\stable_diffusion\models\d.safetensors" --dump_path "C:\Users\Private\Desktop\stable_diffusion\models\d" --from_safetensors
```
