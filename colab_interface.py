"""Install and run the specialized Forge Image and Video Studio on Google Colab.

Copy this file into a Colab code cell or run it with `%run colab_interface.py`.
Secrets are read from Colab Secrets first and environment variables second:

- CIVITAI_API_TOKEN: optional; required for private/early-access Civitai files.
- HF_TOKEN: optional; required for gated Hugging Face video models.

Do not paste API tokens directly into this source file.
"""

import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import requests


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

REPOSITORY_URL = "https://github.com/Stoichiometrical/stable-diffusion-webui-forge"
REPOSITORY_BRANCH = "main"
UPDATE_WEBUI = True

MOUNT_GOOGLE_DRIVE = False
DRIVE_OUTPUT_FOLDER = "forge-image-video-studio"

USE_GRADIO_AUTH = True
GRADIO_USERNAME = "forge"
GRADIO_PASSWORD = ""  # Leave blank to generate a new password each session.

GRADIO_THEME = "remilia/Ghostly"
FORGE_PRESET = "xl"

# Optional files to download before launch. The ID must be a Civitai model-
# version ID, not a model-page ID. Most users can leave this empty and use the
# Download Models tab after Forge starts.
#
# Example:
# INITIAL_CIVITAI_DOWNLOADS = [
#     {"version_id": 1934646, "destination": "checkpoint"},
# ]
INITIAL_CIVITAI_DOWNLOADS = []


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

ROOT_DIR = Path("/content")
REPO_DIR = ROOT_DIR / "stable-diffusion-webui-forge"
MODELS_DIR = REPO_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "Stable-diffusion"
VAE_DIR = MODELS_DIR / "VAE"
LORA_DIR = MODELS_DIR / "Lora"
EMBEDDING_DIR = REPO_DIR / "embeddings"
LOCAL_OUTPUT_DIR = REPO_DIR / "outputs"

DESTINATIONS = {
    "checkpoint": CHECKPOINT_DIR,
    "lora": LORA_DIR,
    "vae": VAE_DIR,
    "embedding": EMBEDDING_DIR,
}


def run(command, cwd=None):
    """Run a command with live output and fail immediately on errors."""
    display_parts = []
    redact_next = False
    for part in command:
        value = str(part)
        display_parts.append("***" if redact_next else value)
        redact_next = value == "--gradio-auth"
    printable = " ".join(display_parts)
    print(f"\n> {printable}")
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def read_secret(name):
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from google.colab import userdata
        return (userdata.get(name) or "").strip()
    except Exception:
        return ""


def install_system_dependencies():
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", "aria2", "ffmpeg", "lz4"])


def install_or_update_repository():
    if not REPO_DIR.exists():
        run([
            "git", "clone", "--depth", "1", "--branch", REPOSITORY_BRANCH,
            REPOSITORY_URL, REPO_DIR,
        ])
        return

    if not (REPO_DIR / ".git").exists():
        raise RuntimeError(f"{REPO_DIR} exists but is not a Git checkout. Move or remove it before continuing.")

    if UPDATE_WEBUI:
        run(["git", "pull", "--ff-only"], cwd=REPO_DIR)
    else:
        print("Existing Forge checkout found; update is disabled.")


def mount_output_directory():
    if not MOUNT_GOOGLE_DRIVE:
        LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return LOCAL_OUTPUT_DIR

    from google.colab import drive
    drive_root = ROOT_DIR / "drive"
    drive.mount(str(drive_root))
    output_dir = drive_root / "MyDrive" / DRIVE_OUTPUT_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def configure_webui(output_dir):
    for directory in DESTINATIONS.values():
        directory.mkdir(parents=True, exist_ok=True)

    config_path = REPO_DIR / "config.json"
    config = read_json(config_path)
    config.update({
        "forge_preset": FORGE_PRESET,
        "gradio_theme": GRADIO_THEME,
        "show_progressbar": True,
        "samples_filename_pattern": "[model_name]_[seed]",
        "outdir_txt2img_samples": str(output_dir / "txt2img-samples"),
        "outdir_img2img_samples": str(output_dir / "img2img-samples"),
        "outdir_txt2img_grids": str(output_dir / "txt2img-grids"),
        "outdir_img2img_grids": str(output_dir / "img2img-grids"),
        "outdir_samples": str(output_dir),
        "outdir_grids": str(output_dir),
    })
    write_json(config_path, config)

    for name in ("txt2img-samples", "img2img-samples", "txt2img-grids", "img2img-grids", "video"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def safe_filename(name):
    value = unquote(str(name or "")).replace("/", "_").replace("\\", "_").strip()
    value = re.sub(r"[^A-Za-z0-9._()\[\] -]+", "_", value).strip(" .")
    if not value:
        raise ValueError("Civitai returned an empty filename.")
    return value


def download_initial_civitai_files(token):
    if not INITIAL_CIVITAI_DOWNLOADS:
        return []

    headers = {"Accept": "application/json", "User-Agent": "Forge-Specialized-Colab/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    downloaded = []
    for item in INITIAL_CIVITAI_DOWNLOADS:
        temporary = None
        version_id = int(item["version_id"])
        destination_name = str(item.get("destination", "checkpoint")).lower()
        if destination_name not in DESTINATIONS:
            raise ValueError(f"Unknown Civitai destination: {destination_name}")

        metadata_response = requests.get(
            f"https://civitai.com/api/v1/model-versions/{version_id}",
            headers=headers,
            timeout=30,
        )
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        files = metadata.get("files") or []
        selected = next((file for file in files if file.get("primary")), files[0] if files else None)
        if selected is None:
            raise RuntimeError(f"Civitai version {version_id} has no downloadable files.")

        filename = safe_filename(item.get("filename") or selected.get("name"))
        target = DESTINATIONS[destination_name] / filename
        if target.exists():
            print(f"Civitai file already exists: {target}")
            downloaded.append(target)
            continue

        url = selected.get("downloadUrl") or metadata.get("downloadUrl")
        if not url:
            url = f"https://civitai.com/api/download/models/{version_id}"
        temporary = target.with_suffix(target.suffix + ".part")
        print(f"Downloading Civitai version {version_id} as {filename}...")
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with open(temporary, "wb") as handle:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            os.replace(temporary, target)
            downloaded.append(target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return downloaded


def configure_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:512"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["PYTHONWARNINGS"] = "ignore"

    civitai_token = read_secret("CIVITAI_API_TOKEN")
    hf_token = read_secret("HF_TOKEN")
    if civitai_token:
        os.environ["CIVITAI_API_TOKEN"] = civitai_token
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    return civitai_token


def launch_webui():
    password = GRADIO_PASSWORD or secrets.token_urlsafe(12)
    args = [
        sys.executable,
        "launch.py",
        "--share",
        "--listen",
        "--lowram",
        "--opt-sub-quad-attention",
        "--no-half-vae",
        "--no-download-sd-model",
        "--disable-extra-extensions",
        "--ckpt-dir", str(CHECKPOINT_DIR),
        "--vae-dir", str(VAE_DIR),
        "--lora-dir", str(LORA_DIR),
        "--embeddings-dir", str(EMBEDDING_DIR),
    ]
    if USE_GRADIO_AUTH:
        args.extend(["--gradio-auth", f"{GRADIO_USERNAME}:{password}"])
        print(f"\nForge login: {GRADIO_USERNAME} / {password}")
    else:
        print("\nWARNING: Gradio authentication is disabled. Anyone with the share URL can use the GPU session.")

    run(args, cwd=REPO_DIR)


def main():
    print("Preparing Forge Image and Video Studio for Google Colab...")
    install_system_dependencies()
    install_or_update_repository()
    configure_environment()
    output_dir = mount_output_directory()
    configure_webui(output_dir)
    downloaded = download_initial_civitai_files(os.environ.get("CIVITAI_API_TOKEN", ""))

    if downloaded:
        config_path = REPO_DIR / "config.json"
        config = read_json(config_path)
        first_checkpoint = next((path for path in downloaded if path.parent == CHECKPOINT_DIR), None)
        if first_checkpoint:
            config["sd_model_checkpoint"] = first_checkpoint.name
            write_json(config_path, config)

    print(f"Outputs: {output_dir}")
    print("Use the Download Models tab to add checkpoints, LoRAs, VAEs, or embeddings by Civitai ID.")
    print("The first video generation downloads its Hugging Face pipeline and may take several minutes.")
    launch_webui()


if __name__ == "__main__":
    main()
