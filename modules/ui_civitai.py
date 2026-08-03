import os
import re
from pathlib import Path
from urllib.parse import unquote

import gradio as gr
import requests

from modules import paths_internal, sd_models, sd_vae, shared


CIVITAI_API = "https://civitai.com/api/v1"
DESTINATIONS = {
    "Checkpoint": lambda: Path(shared.cmd_opts.ckpt_dir or Path(paths_internal.models_path) / "Stable-diffusion"),
    "LoRA": lambda: Path(shared.cmd_opts.lora_dir or Path(paths_internal.models_path) / "Lora"),
    "VAE": lambda: Path(shared.cmd_opts.vae_dir or Path(paths_internal.models_path) / "VAE"),
    "Embedding": lambda: Path(shared.cmd_opts.embeddings_dir),
}


def _headers(token):
    headers = {"Accept": "application/json", "User-Agent": "Forge-Specialized/1.0"}
    token = (token or os.environ.get("CIVITAI_API_TOKEN", "")).strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _safe_filename(name):
    name = unquote(str(name or "")).strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9._()\[\] -]+", "_", name).strip(" .")
    if not name or name in {".", ".."}:
        raise ValueError("Civitai did not provide a usable filename.")
    return name


def _get_json(path, token):
    response = requests.get(f"{CIVITAI_API}/{path}", headers=_headers(token), timeout=30)
    response.raise_for_status()
    return response.json()


def _resolve_version(identifier, identifier_type, token):
    if not str(identifier).strip().isdigit():
        raise ValueError("Enter a numeric Civitai model ID or model-version ID.")

    if identifier_type == "Model ID (latest version)":
        model = _get_json(f"models/{int(identifier)}", token)
        versions = model.get("modelVersions") or []
        if not versions:
            raise ValueError("This model has no downloadable versions.")
        version = versions[0]
        version.setdefault("model", {"name": model.get("name"), "type": model.get("type")})
        return version

    return _get_json(f"model-versions/{int(identifier)}", token)


def _select_file(version, file_id):
    files = version.get("files") or []
    if file_id and str(file_id).strip():
        wanted = str(file_id).strip()
        match = next((item for item in files if str(item.get("id")) == wanted), None)
        if match is None:
            raise ValueError(f"File ID {wanted} is not part of this model version.")
        return match

    primary = next((item for item in files if item.get("primary")), None)
    if primary:
        return primary
    if files:
        return files[0]
    raise ValueError("This model version has no downloadable files.")


def _automatic_destination(version):
    model_type = str((version.get("model") or {}).get("type", "")).lower()
    mapping = {
        "checkpoint": "Checkpoint",
        "lora": "LoRA",
        "locon": "LoRA",
        "vae": "VAE",
        "textualinversion": "Embedding",
        "embedding": "Embedding",
    }
    destination = mapping.get(model_type)
    if not destination:
        raise ValueError(
            f"Civitai type '{model_type or 'unknown'}' is not supported by this specialized UI. "
            "Select a destination manually only if the file is compatible."
        )
    return destination


def _refresh_models(destination):
    if destination == "Checkpoint":
        sd_models.list_models()
    elif destination == "VAE":
        sd_vae.refresh_vae_list()
    elif destination == "Embedding":
        from modules import ui_extra_networks_textual_inversion
        ui_extra_networks_textual_inversion.embedding_db.load_textual_inversion_embeddings(
            force_reload=True,
            sync_with_sd_model=False,
        )
    elif destination == "LoRA":
        try:
            import networks
            networks.list_available_networks()
        except Exception:
            pass


def download_model(identifier, identifier_type, destination, file_id, custom_filename, token, overwrite, allow_pickle, progress=gr.Progress()):
    temp_target = None
    try:
        progress(0, desc="Reading Civitai metadata")
        version = _resolve_version(identifier, identifier_type, token)
        selected = _select_file(version, file_id)
        destination = _automatic_destination(version) if destination == "Automatic" else destination

        filename = _safe_filename(custom_filename or selected.get("name"))
        extension = Path(filename).suffix.lower()
        allowed = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
        if extension not in allowed:
            raise ValueError(f"Unsupported file extension: {extension or '(none)'}")
        if extension != ".safetensors" and not allow_pickle:
            raise ValueError(
                "Non-safetensors files can execute pickle data. Enable 'Allow non-safetensors' "
                "only if you trust this file."
            )

        target_dir = DESTINATIONS[destination]()
        target_dir.mkdir(parents=True, exist_ok=True)
        target = (target_dir / filename).resolve()
        if target.parent != target_dir.resolve():
            raise ValueError("Invalid destination filename.")
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target.name} already exists. Enable overwrite to replace it.")

        download_url = selected.get("downloadUrl") or version.get("downloadUrl")
        if file_id and str(file_id).strip() and not selected.get("downloadUrl"):
            raise ValueError(
                "Civitai did not provide a direct URL for that secondary file. "
                "Download the primary file or enter the secondary file's model-version ID."
            )
        if not download_url:
            download_url = f"https://civitai.com/api/download/models/{version['id']}"

        temp_target = target.with_suffix(target.suffix + ".part")
        with requests.get(download_url, headers=_headers(token), stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(temp_target, "wb") as output:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    output.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        progress(min(downloaded / total, 1.0), desc=f"Downloading {filename}")
        os.replace(temp_target, target)
        _refresh_models(destination)

        size_gib = target.stat().st_size / (1024 ** 3)
        model_name = (version.get("model") or {}).get("name", "Civitai model")
        return (
            f"Downloaded {model_name} to {target} ({size_gib:.2f} GiB). "
            "Use the refresh button beside the model selector if it is not shown immediately."
        )
    except Exception as error:
        return f"Download failed: {type(error).__name__}: {error}"
    finally:
        if temp_target is not None and temp_target.exists():
            try:
                temp_target.unlink()
            except OSError:
                pass


def create_ui():
    with gr.Blocks(analytics_enabled=False) as interface:
        gr.Markdown(
            "## Civitai model downloader\n"
            "Paste either a Civitai **model ID** or an exact **model-version ID**. "
            "Private and early-access files require a Civitai API token. Tokens are used only "
            "for the request and are not saved by this UI."
        )
        with gr.Row():
            identifier = gr.Textbox(label="Civitai ID", placeholder="Example: 1934646", scale=2)
            identifier_type = gr.Radio(
                ["Model version ID", "Model ID (latest version)"],
                value="Model version ID",
                label="ID type",
                scale=2,
            )
            destination = gr.Dropdown(
                ["Automatic", "Checkpoint", "LoRA", "VAE", "Embedding"],
                value="Automatic",
                label="Install as",
                scale=2,
            )
        with gr.Row():
            file_id = gr.Textbox(label="Optional file ID", placeholder="For versions containing multiple files")
            custom_filename = gr.Textbox(label="Optional filename", placeholder="Leave blank to use Civitai's filename")
            token = gr.Textbox(label="Civitai API token", type="password", placeholder="Or set CIVITAI_API_TOKEN")
        with gr.Row():
            overwrite = gr.Checkbox(label="Overwrite an existing file", value=False)
            allow_pickle = gr.Checkbox(label="Allow non-safetensors model files", value=False)
        download_button = gr.Button("Download and install model", variant="primary")
        status = gr.Textbox(label="Download status", interactive=False, lines=3)
        download_button.click(
            fn=download_model,
            inputs=[identifier, identifier_type, destination, file_id, custom_filename, token, overwrite, allow_pickle],
            outputs=[status],
        )
    return interface
