# Forge Image and Video Studio

This Forge checkout is specialized for:

- Text to Image
- Image to Image and core inpainting modes
- Experimental Text to Video
- Experimental Image to Video
- Civitai checkpoint, LoRA, VAE, and embedding downloads

The UI intentionally omits Spaces, Extras, PNG Info, checkpoint merging,
extension installation, ControlNet preprocessors, and unrelated bundled tools.
Their upstream files remain in the repository for import stability, but the
modules are disabled and not exposed to users.

## Civitai downloads

Open **Download Models** and provide:

1. A numeric Civitai model-version ID, or a model-page ID with **Model ID
   (latest version)** selected.
2. **Automatic** destination, or Checkpoint, LoRA, VAE, or Embedding.
3. An API token only for private or early-access files.
4. An optional Civitai file ID when a version contains multiple files.

The downloader selects the primary version file by default. Safetensors files
are accepted by default. Loading pickle-based formats requires explicit
consent. Set `CIVITAI_API_TOKEN` in the environment to avoid entering a token
in the UI.

After downloading, click the refresh button beside Forge's checkpoint, VAE, or
LoRA selector if the item is not shown immediately.

## Video generation

Video support is separate from Forge's image checkpoint loader. Models are
downloaded from Hugging Face on the first job and unloaded after every job to
return GPU and CPU memory to Forge. Starting a video job also fully releases
the current Forge image model; the selected checkpoint is loaded again on the
next image-generation job.

Defaults:

- Text to Video: `cerspense/zeroscope_v2_576w`, 576x320
- Image to Video: `stabilityai/stable-video-diffusion-img2vid-xt`, 1024x576

Both pipelines use CPU offloading and memory-saving chunking. They require a
CUDA GPU and are much slower than image generation. Stable Video Diffusion may
require accepting its Hugging Face license and setting `HF_TOKEN`.

Generated MP4 files are written to `video/` under the configured output root.

## Google Colab

Run `colab_interface.py` in Google Colab. The launcher:

- clones or fast-forward updates the `main` branch;
- installs `aria2`, `ffmpeg`, and `lz4`;
- creates an isolated Python 3.11 environment because this Forge revision and
  its PyTorch pin are not compatible with Colab's Python 3.12 runtime;
- pins compatible Python build tools and disables build isolation for Forge's
  legacy OpenAI CLIP revision;
- creates model and output directories;
- reads `CIVITAI_API_TOKEN` and `HF_TOKEN` from Colab Secrets;
- enables Gradio authentication by default;
- launches with external extensions disabled; and
- preserves an existing checkout instead of deleting it.

Before running it, select **Runtime > Change runtime type > T4 GPU** (or L4/A100).
The launcher stops early with a clear error if Colab has not attached an NVIDIA
GPU.

Add secrets through Colab's key icon instead of putting tokens in source code.
Anyone who previously stored a Civitai token in the launcher should revoke it
and create a replacement.

## Model IDs versus version IDs

A Civitai model page can contain several versions, and each version can contain
several files. The direct download endpoint uses a **model-version ID**. The UI
supports both identifiers, but an exact version ID is the most predictable.

## Current limitations

- Video generation is experimental and has no REST API endpoints yet.
- First use downloads several gigabytes of model weights.
- Free Colab runtimes may run out of CPU RAM or disk space.
- Video cancellation is not implemented.
- Video pipelines reload for each job to minimize persistent memory usage.
- End-to-end video behavior must be validated on a CUDA Colab runtime.

This fork currently uses Diffusers 0.31 and Gradio 4.40. Test upstream merges
carefully because `modules/ui.py`, `modules/launch_utils.py`, and
`modules_forge/config.py` contain specialization changes.
