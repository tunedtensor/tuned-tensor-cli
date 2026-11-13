# Experimental H3-Image LoRA training

`tt pipeline image` trains attention LoRA adapters for
`MiniMaxAI/MiniMax-H3-Image` on local captioned images. It uses the same logged
`uv run --frozen` process pattern as the other TT runners, with a separate
locked image runtime. Existing adapter, decision and foundation specs are
unchanged.

This first integration is **experimental**. It supports text-to-image subject,
style and domain adaptation on one local NVIDIA CUDA GPU with BF16 support.
It does not implement reference-image editing training, quantization,
multi-GPU/remote execution, resume, background jobs, TT model activation,
`tt serve`, or the agent's behavior-spec editor. Use the dedicated command
below; `tt pipeline run` and the text-model commands do not consume image specs.
Unknown fields (including `runtime.gpu`) are rejected.

## Dataset and spec

Create two JSONL manifests. Each row names an image relative to its manifest
and a caption. Training and validation must have distinct image content;
renaming a duplicate does not bypass the split check.

```json
{"image":"images/red-square.png","caption":"A red square on a white background."}
```

Images are converted to RGB after EXIF orientation, then resized and center
cropped to the configured size. Use representative captions for the content
you want the adapter to learn. Empty manifests, malformed rows, duplicate image
bytes and changed input bytes fail before model loading. The actual run also
decodes every image before downloading weights.

Save an image spec, for example `image-training.json`:

```json
{
  "engine": "image",
  "name": "My image style",
  "base_model": "MiniMaxAI/MiniMax-H3-Image",
  "base_model_revision": "5f90473023975e6271b0ed1dd84c4e88e4bbc6d3",
  "text_encoder_revision": "0919fb684102ef115126612f3fbb84d1027e7317",
  "dataset": {
    "training": "data/train.jsonl",
    "validation": "data/validation.jsonl"
  },
  "hyperparameters": {
    "steps": 100,
    "learning_rate": 0.00001,
    "lora_rank": 16,
    "lora_alpha": 16,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": true,
    "width": 512,
    "height": 512,
    "seed": 42
  }
}
```

Both revisions must be immutable Hugging Face commit SHAs. The text encoder is
loaded from `MiniMaxAI/MiniMax-H3`, separately pinned; it is never inferred from
the image model revision. Dataset paths resolve relative to the spec.

```bash
tt pipeline image --spec image-training.json --dry-run
tt pipeline image --spec image-training.json --output ./image-run --verbose
```

The dry-run reads and hashes inputs without downloading models, initializing
Python or creating a run directory. The real run downloads missing pinned
components into the normal Hugging Face cache. Use `HF_HOME` to locate the
cache. The full model and its text encoder require roughly 135 GB of weight
downloads plus runtime, cache and working space. Loading is sequential: encode
captions, unload the text encoder, encode images, unload the VAE, load the
transformer. The frozen BF16 transformer alone is roughly 66 GB; training needs
additional memory. This is not a 24 GB GPU training recipe. The 512-pixel default
keeps smoke runs manageable; it is not an established quality recipe for the
native 4-megapixel to 20-megapixel generation range.

Each optimizer step averages `gradient_accumulation_steps` single-image
microbatches. Only the main transformer's attention query/key/value/output
LoRA weights train; the encoder, VAE, base transformer and six output heads
remain frozen. The objective samples one of the six released noise levels,
selects its corresponding PDD output head and predicts **clean minus noise**
at timestep **one minus sigma**. It uses upstream Diffusers' H3 packing and
positional encoding. Generic language-model LoRA training is not used here.

## Results and export

A new output directory contains:

- `config.json`: resolved spec, source hashes and image/caption records.
- `training.log`: complete subprocess output, including model-load failures.
- `cache/`: frozen caption embeddings and normalized image latents.
- `adapter/`: standard PEFT `adapter_config.json`, `adapter_model.safetensors`
  and `tt_image_adapter.json` with pinned model provenance.
- `report.json`: baseline/candidate held-out velocity MSE, optimizer steps,
  adapter checksum, hardware and reload verification.

A failed run writes `failure.json` and exits unsuccessfully. Existing output
directories are refused; retry into a new directory. A successful optimizer
run is not a quality certification. Lower held-out velocity MSE is a useful
training diagnostic, not proof of better generated images; `quality_validated`
remains false. Compare generated images with fixed prompts/seeds at the desired
resolution before using an adapter.

The candidate is evaluated on every held-out image at all six noise levels,
using the same seeded noise as baseline. After export, live adapter weights are
cleared and the saved tensors are loaded and evaluated again. A mismatched
result or checksum makes the run fail.

To use an exported adapter with the pinned Diffusers runtime, construct the
same H3-Image modular pipeline and load the transformer at the recorded image
revision, then inject the PEFT adapter before generation:

```python
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from safetensors.torch import load_file

# pipe is a loaded MiniMaxH3ImageModularPipeline with the recorded base weights.
adapter_dir = "./image-run/adapter"
inject_adapter_in_model(LoraConfig.from_pretrained(adapter_dir), pipe.transformer)
for name, parameter in pipe.transformer.named_parameters():
    if "lora_" in name:
        parameter.data = parameter.data.float()  # Preserve the saved adapter precision.
result = set_peft_model_state_dict(
    pipe.transformer,
    load_file(f"{adapter_dir}/adapter_model.safetensors"),
)
assert not result.unexpected_keys
assert not any("lora_" in key for key in result.missing_keys)
pipe.transformer.eval()
```

Keep the adapter's recorded base and text-encoder revisions when loading the
other pipeline components too. See the [official model card](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
and [Diffusers pipeline documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3_image)
for generation/offloading. The runtime pins a Diffusers commit because the H3
modular APIs are new; changing that pin requires rerunning training/export tests.

## Validation

`npm run test:image` exercises the real H3 transformer architecture at reduced
dimensions on CPU, including frozen-weight preservation, gradient accumulation,
gradient checkpointing, all six head/timestep mappings and export into a fresh
transformer. `TT_IMAGE_TEST_DEVICE=cuda npm run test:image` runs the same checks
on CUDA. These tests use synthetic weights; they establish mechanics, not
pretrained-model quality. TypeScript runtime tests cover source validation,
side-effect-free previews, split leakage, failed execution, output collisions
and adapter checksum verification.
