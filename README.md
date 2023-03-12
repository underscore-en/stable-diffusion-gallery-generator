# stable diffusion tactical reload

This script allows you to reload the arguments at runtime using diffuser API. 

### generate.py

```bash
py ./generate.py --model_path "XpucT/Deliberate" --prompt_file_path "./config/prompt.txt" --negative_prompt_file_path "./config/negative_prompt.txt" --guidance_scale_file_path  "./config/guidance_scale.txt" --dimension_file_path "./config/dimension.txt" --num_inference_steps_file_path "./config/num_inference_steps.txt" --dump_path "C:\Users\Private\Desktop\sd_output"
```

| arg | Description |
| ----------- | -----------                                       |
| model path | path to model, can be local diffuser format or hugging face id |
| prompt file path | txt file, can be multiline that will be joined by comma |
| negative prompt file path | txt file, can be multiline that will be joined by comma |
| guidance scale file path | txt file, single line integer |
| dimension file path | txt file, 2 line int width then height |
| num_inference_steps_file_path | txt file, single line integer |
| dump_path | path of output folder, no trailing slash |
