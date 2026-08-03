# Specialized Forge Handoff

> **Applied in the actual repository on August 3, 2026.** The UI,
> extension filtering, Civitai downloader, experimental video interface,
> dependency changes, Colab launcher, and specialized documentation described
> below have now been implemented. Static validation passed; CUDA/Colab
> end-to-end generation tests remain pending.

> **Colab compatibility fix:** Current Colab reported Python 3.12 at
> `/usr/bin/python3`, while this Forge revision supports Python through 3.11
> and pins PyTorch 2.3.1. The launcher now installs `uv`, creates an isolated
> Python 3.11 environment at `/content/forge-python-3.11`, and launches Forge
> from it. It also enables live Forge installer output for easier diagnosis.

> **Additional Colab fixes:** The launcher now verifies that an NVIDIA GPU is
> attached before installation. It pins pip/setuptools build tooling and uses
> the venv's setuptools for Forge's legacy OpenAI CLIP revision, avoiding the
> incompatible isolated build that fails on current Colab. Video startup now
> fully releases Forge's image model from CPU RAM and invalidates its model
> cache so the selected checkpoint reloads on the next image job.
> The Civitai downloader was also checked against the live public API. Its
> automatic destination now considers a selected secondary file's type, so a
> VAE attached to a checkpoint version is installed in `models/VAE` rather
> than `models/Stable-diffusion`.
> Current Colab also exports a notebook-only `MPLBACKEND` value that is not
> available inside the isolated Forge venv. The launcher now selects the
> headless Matplotlib `Agg` backend before starting Forge.
> The legacy Postprocessing settings are now hidden while their defaults remain
> registered. This prevents Settings from iterating the uninitialized Extras
> postprocessing script list in the specialized UI.

## Purpose of this document

This document is the handoff for reproducing the work in the actual
`Stoichiometrical/stable-diffusion-webui-forge` repository.

The current workspace was a copied checkout used for analysis and prototyping.
The changes described here need to be applied, reviewed, tested on Google
Colab, committed, and pushed in the actual repository.

## User's requested outcome

Specialize Stable Diffusion WebUI Forge for only these workflows:

1. Text to Image
2. Image to Image and closely related image-generation functionality
3. Text to Video, if practical on Google Colab
4. Image to Video, if practical on Google Colab
5. Easy Civitai model installation from the WebUI using a model ID or
   model-version ID

Remove or disable unrelated functionality and make the interface easier for a
non-technical user. The application will primarily be installed and launched
through `colab_interface.py` on Google Colab.

## Important findings from the original repository

- The repository is Stable Diffusion WebUI Forge, not a purpose-built video
  application.
- Native Forge supports text-to-image and image-to-image.
- Forge's `img2img` is image-to-image, not image-to-video.
- No real text-to-video or image-to-video generator existed in the inspected
  checkout.
- The existing video-related source matches were background-removal utilities,
  model comments, upscaler names, or generic neural-network code.
- Native API routes included `/sdapi/v1/txt2img` and
  `/sdapi/v1/img2img`, but no video-generation endpoints.
- Hiding tabs alone is insufficient because the original `modules/ui.py`
  constructs most interfaces before applying the hidden-tab filter.
- Directly deleting large parts of Forge is risky because imports, shared
  globals, script callbacks, settings, model loading, and extensions are
  tightly coupled.

The chosen approach is therefore to stop constructing unrelated interfaces and
disable unrelated built-in extensions while retaining their upstream source
where removal could make future merges or imports fragile.

## Prototype implementation already completed

The copied checkout currently contains a working prototype of the following
changes. These changes are not a substitute for applying and testing them in
the actual repository.

### Specialized top-level tabs

`stable-diffusion-webui-forge/modules/ui.py` was changed so the top-level UI
contains only:

- `Text to Image`
- `Image to Image`
- `Video`
- `Download Models`
- `Settings`

The following original interfaces are no longer constructed or rendered:

- Spaces
- Extras
- PNG Info
- Checkpoint Merger
- Extensions

Extension-provided top-level tabs are also not appended through
`script_callbacks.ui_tabs_callback()` in the specialized interface.

The application title was changed to `Forge Image and Video Studio`.

Do not remove `Settings`: model behavior, output paths, memory options, and
other generation settings still depend on it.

### Disabled bundled extensions

`stable-diffusion-webui-forge/modules_forge/config.py` was expanded so
`always_disabled_extensions` disables unrelated Spaces, preprocessors,
ControlNet add-ons, upscalers, visual modifiers, and experimental utilities.

The intended minimal enabled built-ins are effectively:

- `extra-options-section`
- `mobile`
- `prompt-bracket-checker`
- `sd_forge_lora`

LoRA support must remain because it is directly relevant to image generation
and to Civitai downloads.

The disabled list in the prototype contains:

```text
sd-webui-controlnet
multidiffusion-upscaler-for-automatic1111
forge_legacy_preprocessors
forge_preprocessor_inpaint
forge_preprocessor_marigold
forge_preprocessor_normalbae
forge_preprocessor_recolor
forge_preprocessor_reference
forge_preprocessor_revision
forge_preprocessor_tile
forge_space_animagine_xl_31
forge_space_birefnet
forge_space_example
forge_space_florence_2
forge_space_geowizard
forge_space_iclight
forge_space_idm_vton
forge_space_illusion_diffusion
forge_space_photo_maker_v2
forge_space_sapiens_normal
ScuNET
sd_forge_controlllite
sd_forge_controlnet
sd_forge_dynamic_thresholding
sd_forge_fooocus_inpaint
sd_forge_freeu
sd_forge_ipadapter
sd_forge_kohya_hrfix
sd_forge_latent_modifier
sd_forge_multidiffusion
sd_forge_neveroom
sd_forge_perturbed_attention
sd_forge_sag
sd_forge_stylealign
soft-inpainting
SwinIR
```

Review this list against the actual repository because its built-in extension
names may have changed.

### Prevent disabled extension installers from running

The runtime extension loader already considered `always_disabled_extensions`,
but the built-in extension installer path did not.

In `modules/launch_utils.py`, `list_extensions_builtin()` was changed from:

```python
disabled_extensions = set(settings.get('disabled_extensions', []))
```

to:

```python
disabled_extensions = set(
    settings.get('disabled_extensions', []) + always_disabled_extensions
)
```

This is important on Colab. Otherwise disabled heavy extensions such as
ControlNet or legacy preprocessors may still run their installers even though
they are not used at runtime.

## Civitai model downloader

A new core module was added:

```text
stable-diffusion-webui-forge/modules/ui_civitai.py
```

It creates the `Download Models` tab and supports:

- Civitai model-version IDs
- Civitai model-page IDs, selecting the latest version
- Automatic destination based on Civitai model type
- Explicit destinations:
  - Checkpoint
  - LoRA
  - VAE
  - Embedding
- Optional file ID for versions with multiple files
- Optional custom filename
- Optional Civitai API token
- Environment fallback through `CIVITAI_API_TOKEN`
- Streaming downloads with Gradio progress
- Temporary `.part` files followed by atomic replacement
- Filename sanitization and destination-path validation
- Existing-file protection unless overwrite is explicitly enabled
- Automatic refresh of Forge checkpoint, VAE, embedding, or LoRA registries

### Civitai terminology

A model page contains multiple model versions, and a version can contain
multiple files. Civitai's direct download endpoint uses a model-version ID:

```text
https://civitai.com/api/download/models/{modelVersionId}
```

The UI must clearly distinguish `Model version ID` from
`Model ID (latest version)`.

Metadata endpoints used:

```text
GET https://civitai.com/api/v1/models/{modelId}
GET https://civitai.com/api/v1/model-versions/{modelVersionId}
```

Civitai documentation consulted:

```text
https://github.com/civitai/civitai/wiki/REST-API-Reference
```

The wiki now points toward Civitai's newer developer documentation, so verify
the endpoints when applying the work in the actual repository.

### Civitai security rules

- Do not hard-code an API token.
- Use a password-type textbox and/or `CIVITAI_API_TOKEN`.
- Never print a token or a token-bearing URL.
- Default to `.safetensors`.
- Require explicit consent for `.ckpt`, `.pt`, `.pth`, or `.bin` because these
  may contain pickle data.
- Keep Forge safe-unpickle checks enabled.
- Restrict all download URLs to metadata obtained from Civitai and restrict
  destinations to known Forge model directories.
- Remove incomplete `.part` files after failed downloads.

### Exposed key in the original Colab code

The supplied original `colab_interface.py` contained a plaintext Civitai API
key. It was removed from the prototype.

The user must revoke that old key and create a replacement even if the file was
never intentionally published. Do not restore the old key from history or from
the conversation.

## Experimental video support

A new core module was added:

```text
stable-diffusion-webui-forge/modules/ui_video.py
```

It creates one `Video` top-level tab with:

- `Text to Video`
- `Image to Video`

Video models are deliberately separate from Forge's image checkpoint loader.
They are loaded lazily through Diffusers only when a video job starts.

Before a video job:

1. Ensure CUDA is available.
2. Ask Forge to unload active image-model weights.
3. Run Python garbage collection.
4. Clear unused CUDA memory.

After every video job:

1. Free Diffusers model hooks.
2. Delete the video pipeline.
3. Run garbage collection.
4. Clear CUDA memory.

This makes repeated jobs slower but is safer for free or low-memory Colab
runtimes. A process-level lock prevents two video jobs from running at once.

Generated videos are saved under the configured Forge output root in:

```text
video/
```

### Text-to-video prototype

Default model:

```text
cerspense/zeroscope_v2_576w
```

Default output resolution:

```text
576x320
```

Pipeline behavior:

- Uses `diffusers.DiffusionPipeline`
- Uses `DPMSolverMultistepScheduler`
- Uses FP16
- Enables model CPU offload
- Enables VAE slicing
- Enables UNet forward chunking
- Exposes prompt, negative prompt, frames, steps, guidance, seed, and FPS
- Reads a gated/private token from the textbox or `HF_TOKEN`

Diffusers 0.31 documentation used:

```text
https://huggingface.co/docs/diffusers/v0.31.0/api/pipelines/text_to_video
```

### Image-to-video prototype

Default model:

```text
stabilityai/stable-video-diffusion-img2vid-xt
```

Pipeline behavior:

- Uses `StableVideoDiffusionPipeline`
- Uses FP16 variant weights
- Fits the uploaded source image to 1024x576
- Enables model CPU offload
- Enables UNet forward chunking
- Uses a small configurable VAE decode chunk
- Exposes frames, steps, motion strength, noise/variation, seed, and FPS
- Reads a gated/private token from the textbox or `HF_TOKEN`

Stable Video Diffusion is image-conditioned and does not take a normal text
prompt in this implementation. The interface must state this clearly.

Diffusers 0.31 documentation used:

```text
https://huggingface.co/docs/diffusers/v0.31.0/en/using-diffusers/svd
```

The documentation states that CPU offload, forward chunking, and small decode
chunks can reduce SVD VRAM usage below 8 GB, although actual Colab RAM, model
download time, and execution time still need real validation.

### Video limitations that must remain visible

- Video generation is experimental.
- It requires a CUDA runtime.
- First use downloads several gigabytes from Hugging Face.
- It is much slower than image generation.
- Hugging Face licenses may need to be accepted before gated weights download.
- Free Colab CPU RAM or disk may still be insufficient in some sessions.
- Pipelines are unloaded after each job, so subsequent generation reloads them
  from the Hugging Face cache.
- Actual cancellation is not yet implemented.
- There are no new REST video endpoints yet; video exists only in the Gradio UI
  prototype.

Do not claim production-ready video support until it is tested on at least a
Colab T4 runtime.

## Dependency changes

The prototype changed `requirements_versions.txt`:

```text
Pillow>=9.5,<12
imageio==2.34.2
imageio-ffmpeg==0.5.1
sentencepiece==0.2.0
```

Existing important pins in the inspected checkout were:

```text
gradio==4.40.0
diffusers==0.31.0
accelerate==0.31.0
transformers==4.46.1
huggingface-hub==0.26.2
```

The Pillow pin was relaxed because the original Colab script worked around a
Pillow incompatibility by uninstalling Forge's version and forcing Pillow
11.2.1 after installation. That workaround was brittle and removed.

In the actual repository, first inspect its current pins. If it has newer
Gradio or Diffusers versions, adapt the code to their current APIs instead of
downgrading them to the prototype versions.

## Colab launcher changes

The workspace-level `colab_interface.py` was completely refactored.

The original file had these problems:

- A plaintext Civitai API key
- Deleted the entire repository on every run
- A configured update flag that was not meaningfully used
- Selected a checkpoint filename that it never downloaded
- Forced Pillow after dependency installation
- Disabled safe unpickling
- Enabled insecure extension access even though it was unnecessary
- Launched a public Gradio share without authentication by default
- Mixed model IDs and model-version IDs
- Used shell-string construction through `get_ipython()`
- Included unused downloader configuration and imports
- Contained visibly corrupted emoji/text encoding

The refactored launcher:

- Uses standard Python and `subprocess.run()` argument lists
- Clones the specialized repository only when missing
- Uses `git pull --ff-only` when updating an existing checkout
- Never automatically deletes the checkout
- Uses the `main` branch in the inspected repository
- Installs `aria2`, `ffmpeg`, and `lz4`
- Creates checkpoint, VAE, LoRA, embedding, output, and video directories
- Supports optional Google Drive output persistence
- Reads `CIVITAI_API_TOKEN` and `HF_TOKEN` from Colab Secrets first and
  environment variables second
- Enables generated Gradio authentication by default
- Prints the generated login without printing API tokens
- Keeps Forge safe model loading enabled
- Launches with user-installed extensions disabled
- Supports an optional list of initial Civitai model-version downloads
- Encourages most users to download models from the WebUI after launch
- Uses a correctly downloaded checkpoint filename if an initial checkpoint was
  configured

### Expected Colab secrets

Add these using Colab's key/secrets icon:

```text
CIVITAI_API_TOKEN
HF_TOKEN
```

Both are optional for public assets. `HF_TOKEN` may be necessary for Stable
Video Diffusion after accepting the model license.

### Expected repository configuration

The prototype launcher currently uses:

```python
REPOSITORY_URL = "https://github.com/Stoichiometrical/stable-diffusion-webui-forge"
REPOSITORY_BRANCH = "main"
```

Confirm the actual repository's default branch. The Colab launcher will not see
local changes until they are committed and pushed to this URL and branch.

### Colab launch arguments in the prototype

```text
--share
--listen
--lowram
--opt-sub-quad-attention
--no-half-vae
--no-download-sd-model
--disable-extra-extensions
--ckpt-dir <checkpoint directory>
--vae-dir <VAE directory>
--lora-dir <LoRA directory>
--embeddings-dir <embedding directory>
--gradio-auth <generated credentials>
```

Verify each flag against the actual repository's current argument parser.

## Files changed or added in the prototype

Inside the Forge checkout:

```text
M modules/launch_utils.py
M modules/ui.py
M modules_forge/config.py
M requirements_versions.txt
A modules/ui_civitai.py
A modules/ui_video.py
A SPECIALIZED_README.md
```

At the workspace level:

```text
M README.md
M colab_interface.py
A ToChange.md
```

The workspace-level `README.md` contains the original architectural analysis.
`SPECIALIZED_README.md` contains user-facing instructions for the specialized
application.

## Validation already performed on the prototype

The following checks passed:

```text
python -m py_compile colab_interface.py
python -m py_compile modules/ui_civitai.py
python -m py_compile modules/ui_video.py
python -m py_compile modules/ui.py
python -m py_compile modules/launch_utils.py
python -m py_compile modules_forge/config.py
git diff --check
```

A search confirmed the previously embedded API token was no longer present in
the modified Colab or new UI modules.

No end-to-end runtime test was possible in the prototype environment because it
did not have:

- Forge's pinned Gradio environment
- CUDA
- Image checkpoints
- Downloaded video models

Syntax validation is not sufficient. Full testing remains mandatory in the
actual repository and on Colab.

## Steps to perform in the actual repository

### 1. Inspect before editing

Run:

```text
git status --short
git branch --show-current
git remote -v
git log -1 --oneline
rg --files -g AGENTS.md
```

Read any `AGENTS.md` instructions before changing files. Preserve all existing
user changes and do not overwrite a dirty worktree.

Compare the actual versions of:

- `modules/ui.py`
- `modules/launch_utils.py`
- `modules/extensions.py`
- `modules_forge/config.py`
- `requirements_versions.txt`
- `modules/sd_models.py`
- Gradio
- Diffusers
- Transformers
- Hugging Face Hub

Do not blindly copy prototype code if upstream structure or APIs differ.

### 2. Reapply the specialization

- Stop constructing Spaces, Extras, PNG Info, Checkpoint Merger, and Extensions
  in `modules/ui.py`.
- Keep Text to Image, Image to Image, Settings, and Forge's core model manager.
- Add core Civitai and Video interfaces.
- Ensure model-selection quick settings remain accessible.
- Ensure txt2img/img2img paste fields and output panels do not reference removed
  interfaces in a way that causes runtime errors.
- Decide whether extension-provided top-level tabs must be entirely suppressed
  or filtered through an allowlist.

### 3. Reapply extension filtering

- Add or update `always_disabled_extensions`.
- Ensure both runtime loading and built-in installer discovery respect the same
  list.
- Keep LoRA enabled.
- Confirm disabled extension names against the real directories.

### 4. Add the Civitai interface

- Port `modules/ui_civitai.py`.
- Confirm the current Civitai metadata schema.
- Test public downloads without a token.
- Test authenticated downloads with a temporary token from Colab Secrets.
- Test checkpoint, LoRA, VAE, and embedding destinations.
- Test duplicate protection, overwrite, secondary file selection, failed
  downloads, and `.part` cleanup.
- Verify a downloaded item appears after refreshing the corresponding Forge
  selector.

### 5. Add the video interface

- Port `modules/ui_video.py` against the actual Diffusers version.
- Verify Zeroscope still loads with that version.
- Verify SVD-XT still loads with that version.
- Confirm CPU-offload APIs and `token`/authentication keyword names.
- Confirm video encoding works through `imageio-ffmpeg` and installed `ffmpeg`.
- Test switching from Forge image generation to video and back to image
  generation without stale model state or VRAM exhaustion.
- Confirm generated video paths respect Google Drive output configuration.

### 6. Port and test the Colab launcher

- Place or update `colab_interface.py` where the user expects to run it.
- Confirm repository URL and branch.
- Test a clean Colab installation.
- Test rerunning in the same session without deletion.
- Test Google Drive mounted and unmounted modes.
- Test with no initial model so the Download Models tab can bootstrap the app.
- Test with one initial Civitai checkpoint version.
- Confirm public share requires authentication.
- Confirm secrets never appear in cell output or exception messages.

### 7. Run local and Colab validation

At minimum:

```text
python -m py_compile <all modified Python files>
git diff --check
```

Then launch Forge in UI debug mode if available, followed by a normal launch.

Functional test matrix:

| Test | Expected result |
|---|---|
| Launch with no checkpoint | UI opens and Download Models is usable |
| Download public checkpoint | File saved and checkpoint list refreshes |
| Download LoRA | File saved and LoRA list refreshes |
| Reject unsafe file by default | Non-safetensors download is blocked |
| Txt2img | Generates and saves an image |
| Img2img | Generates from uploaded image |
| Inpainting | Core img2img inpainting remains functional |
| Txt2video | Generates MP4 on a Colab GPU |
| Img2video | Generates MP4 from uploaded image |
| Image after video | Forge reloads and generates without restart |
| Video after image | Image weights unload and video runs without OOM |
| Drive outputs | Images and video persist in Drive |
| Rerun launcher | Existing checkout is preserved and updated safely |

Test on a Colab T4 first. If available, repeat on L4 and A100.

## Potential issues to watch closely

### Gradio API differences

The prototype used Gradio 4.40 conventions such as `gr.Blocks`, `gr.Video`,
`gr.Progress`, and positional event inputs/outputs. Adjust if the actual
repository uses a newer Gradio version.

### Hugging Face authentication keyword

The prototype passes `token=` to `from_pretrained()`. Older libraries sometimes
used `use_auth_token=`. Use the keyword required by the actual Hugging Face Hub
version.

### Forge model unloading

The prototype calls `sd_models.unload_model_weights()` before video. Confirm
that the actual Forge checkout can reload/offload its model correctly after a
Diffusers video job. Pay particular attention to Forge's model hash and cached
`model_data.sd_model` state.

### CPU RAM pressure

GPU offload moves weights into CPU RAM. Loading SVD and an SDXL/Flux checkpoint
in the same process may exceed free Colab system RAM even if GPU VRAM is safe.
The prototype unloads the video pipeline after every job, but Forge image-model
objects may still occupy CPU memory. Measure actual memory usage.

If this remains unstable, the next design should run video generation in a
separate subprocess so the operating system can reclaim all video memory when
the process exits.

### Output path behavior

The video module must use Forge's configured output root, not a hard-coded
`/content` path, so Google Drive persistence works.

### Civitai secondary files

A model version can expose more than one file. If metadata does not provide a
direct `downloadUrl` for a selected secondary file, do not silently download
the primary file instead. Return a clear error or implement Civitai's current
file-selection query parameters.

### Public Gradio URLs

Never disable authentication by default when `--share` is enabled. Anyone with
the URL could otherwise consume the Colab GPU and potentially invoke dangerous
functionality.

## Recommended scope boundary

For the first working release, keep:

- Core Forge checkpoint loading
- Text to Image
- Image to Image and inpainting
- LoRAs, VAEs, and embeddings
- Core samplers and schedulers
- Output saving
- Essential settings
- Civitai downloads
- Experimental text-to-video and image-to-video

Do not add yet:

- Video-to-video
- ControlNet
- Training
- Checkpoint merging
- Extension installation
- Spaces
- Face restoration tabs
- PNG inspection
- Multiple video backends
- Complex video upscaling
- REST video endpoints

These can be evaluated only after the minimal Colab workflow is reliable.

## Completion criteria

The work is complete only when:

1. The specialized repository is committed and pushed to the configured branch.
2. A clean Colab runtime can clone and launch it.
3. The share URL is authenticated.
4. A user can install a public Civitai checkpoint from the UI using only an ID.
5. Txt2img and img2img work with that checkpoint.
6. At least one text-to-video job completes on Colab and produces an MP4.
7. At least one image-to-video job completes on Colab and produces an MP4.
8. Image generation still works after a video job without restarting Colab.
9. No API token is present in tracked source or printed output.
10. Documentation explains model IDs, secrets, downloads, output persistence,
    video limitations, and the first-run delay.
