# stable-diffusion-gallery-generator

Generate a variety of galleries with stable diffusion.

### `./generate.py`

```python
parser.add_argument("--model_path", required=True, type=str)
parser.add_argument("--prompt_file_path", required=True, type=str)
parser.add_argument("--negative_prompt_file_path", required=True, type=str)
parser.add_argument("--gallery_dump_path", required=True, type=str)
parser.add_argument("--job_file_path", required=True, type=str)
```