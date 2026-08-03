import gc
import os
import threading
import time
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

from modules import paths_internal, sd_models, shared
from modules.call_queue import wrap_queued_call


VIDEO_LOCK = threading.Lock()
DEFAULT_T2V_MODEL = "cerspense/zeroscope_v2_576w"
DEFAULT_I2V_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"


def _output_path(prefix):
    output_root = Path(shared.opts.outdir_samples) if shared.opts.outdir_samples else Path(paths_internal.data_path) / "outputs"
    output_dir = output_root / "video"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}.mp4")


def _token(value):
    return (value or os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")).strip() or None


def _prepare_gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Video generation requires a CUDA GPU. In Colab, select a GPU runtime first.")
    sd_models.unload_model_weights()
    gc.collect()
    torch.cuda.empty_cache()
    return torch


def _cleanup(pipe=None):
    if pipe is not None:
        try:
            pipe.maybe_free_model_hooks()
        except Exception:
            pass
        del pipe
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def text_to_video(prompt, negative_prompt, model_id, frames, steps, guidance, seed, fps, hf_token, progress=gr.Progress()):
    if not (prompt or "").strip():
        return None, "Enter a prompt first."
    if not VIDEO_LOCK.acquire(blocking=False):
        return None, "Another video job is already running."

    pipe = None
    try:
        torch = _prepare_gpu()
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
        from diffusers.utils import export_to_video

        progress(0.05, desc="Loading text-to-video model")
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        token = _token(hf_token)
        if token:
            kwargs["token"] = token
        pipe = DiffusionPipeline.from_pretrained((model_id or DEFAULT_T2V_MODEL).strip(), **kwargs)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "unet"):
            pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        progress(0.15, desc="Generating frames")
        result = pipe(
            prompt=(prompt or "").strip(),
            negative_prompt=(negative_prompt or "").strip() or None,
            num_frames=int(frames),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            height=320,
            width=576,
            generator=generator,
        )
        progress(0.9, desc="Encoding MP4")
        path = _output_path("txt2video")
        export_to_video(result.frames[0], path, fps=int(fps))
        return path, f"Saved to {path}. The video model was unloaded to return memory to Forge."
    except Exception as error:
        return None, f"Video generation failed: {type(error).__name__}: {error}"
    finally:
        _cleanup(pipe)
        VIDEO_LOCK.release()


def image_to_video(image, model_id, frames, steps, motion, noise, seed, fps, decode_chunk_size, hf_token, progress=gr.Progress()):
    if image is None:
        return None, "Upload a source image first."
    if not VIDEO_LOCK.acquire(blocking=False):
        return None, "Another video job is already running."

    pipe = None
    try:
        torch = _prepare_gpu()
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import export_to_video

        progress(0.05, desc="Loading image-to-video model")
        kwargs = {"torch_dtype": torch.float16, "variant": "fp16", "low_cpu_mem_usage": True}
        token = _token(hf_token)
        if token:
            kwargs["token"] = token
        pipe = StableVideoDiffusionPipeline.from_pretrained((model_id or DEFAULT_I2V_MODEL).strip(), **kwargs)
        pipe.enable_model_cpu_offload()
        pipe.unet.enable_forward_chunking()

        source = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
        source = ImageOps.fit(source, (1024, 576), method=Image.Resampling.LANCZOS)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        progress(0.15, desc="Animating image")
        result = pipe(
            source,
            num_frames=int(frames),
            num_inference_steps=int(steps),
            motion_bucket_id=int(motion),
            noise_aug_strength=float(noise),
            decode_chunk_size=int(decode_chunk_size),
            generator=generator,
        )
        progress(0.9, desc="Encoding MP4")
        path = _output_path("img2video")
        export_to_video(result.frames[0], path, fps=int(fps))
        return path, f"Saved to {path}. The video model was unloaded to return memory to Forge."
    except Exception as error:
        return None, f"Video generation failed: {type(error).__name__}: {error}"
    finally:
        _cleanup(pipe)
        VIDEO_LOCK.release()


def create_ui():
    with gr.Blocks(analytics_enabled=False) as interface:
        gr.Markdown(
            "## Video generation (experimental)\n"
            "Video weights are downloaded from Hugging Face on first use and unloaded after every job. "
            "This saves Colab memory but makes startup slower. Image-to-video uses the uploaded image "
            "as its condition and does not accept a text prompt."
        )
        with gr.Tabs():
            with gr.Tab("Text to Video"):
                t2v_prompt = gr.Textbox(label="Prompt", lines=3)
                t2v_negative = gr.Textbox(label="Negative prompt", lines=2)
                t2v_model = gr.Textbox(label="Hugging Face model", value=DEFAULT_T2V_MODEL)
                with gr.Row():
                    t2v_frames = gr.Slider(8, 48, value=24, step=1, label="Frames")
                    t2v_steps = gr.Slider(5, 50, value=25, step=1, label="Steps")
                    t2v_guidance = gr.Slider(1, 20, value=9, step=0.5, label="Guidance")
                    t2v_fps = gr.Slider(4, 24, value=8, step=1, label="FPS")
                    t2v_seed = gr.Number(value=42, precision=0, label="Seed")
                t2v_token = gr.Textbox(label="Optional Hugging Face token", type="password", placeholder="Or set HF_TOKEN")
                t2v_button = gr.Button("Generate text-to-video", variant="primary")
                t2v_output = gr.Video(label="Generated video")
                t2v_status = gr.Textbox(label="Status", interactive=False)
                t2v_button.click(
                    wrap_queued_call(text_to_video),
                    [t2v_prompt, t2v_negative, t2v_model, t2v_frames, t2v_steps, t2v_guidance, t2v_seed, t2v_fps, t2v_token],
                    [t2v_output, t2v_status],
                )

            with gr.Tab("Image to Video"):
                i2v_image = gr.Image(label="Source image", type="pil")
                i2v_model = gr.Textbox(label="Hugging Face model", value=DEFAULT_I2V_MODEL)
                with gr.Row():
                    i2v_frames = gr.Slider(14, 25, value=25, step=1, label="Frames")
                    i2v_steps = gr.Slider(5, 50, value=25, step=1, label="Steps")
                    i2v_motion = gr.Slider(1, 255, value=127, step=1, label="Motion strength")
                    i2v_noise = gr.Slider(0, 1, value=0.02, step=0.01, label="Image variation")
                    i2v_fps = gr.Slider(4, 24, value=7, step=1, label="FPS")
                    i2v_seed = gr.Number(value=42, precision=0, label="Seed")
                    i2v_decode = gr.Slider(1, 8, value=2, step=1, label="Decode chunk")
                i2v_token = gr.Textbox(label="Optional Hugging Face token", type="password", placeholder="Or set HF_TOKEN")
                i2v_button = gr.Button("Generate image-to-video", variant="primary")
                i2v_output = gr.Video(label="Generated video")
                i2v_status = gr.Textbox(label="Status", interactive=False)
                i2v_button.click(
                    wrap_queued_call(image_to_video),
                    [i2v_image, i2v_model, i2v_frames, i2v_steps, i2v_motion, i2v_noise, i2v_seed, i2v_fps, i2v_decode, i2v_token],
                    [i2v_output, i2v_status],
                )
    return interface
