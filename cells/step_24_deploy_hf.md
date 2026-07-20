# Step 24 — HF Hub + Spaces deployment

Implements `docs/modules/deploy_env_space.md` §8.2 and DESIGN.md §11.3, §11.4 deliverables: push the LoRA adapter, env Space, demo Space, and Indic-briefs dataset to Hugging Face.

Four push helpers, all using the **new** `hf upload` CLI per `deploy_env_space.md §8.2`. Calling the deprecated `huggingface-cli` raises `DeprecatedCliError` so the bug never sneaks in.

`push_lora_to_hub(checkpoint_path, repo_id, token)` pushes adapter-only artifacts (`adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`, `README.md`) with `safe_serialization=True`. Naive 4-bit → 16-bit merging is forbidden — `merge_4bit_to_16bit=True` raises `NaiveMergeForbiddenError` (DESIGN.md §10.5, CLAUDE.md §13).

`push_env_space(repo_id, token, space_dir=...)` pushes the Docker-based env Space (CPU basic per `deploy_env_space.md §6.3`). `push_demo_space(repo_id, token, hardware="zero-gpu" | "a10g-small")` pushes the Gradio demo Space, defaulting to ZeroGPU and supporting the A10G fallback per `deploy_demo_space.md §3.1`. `push_dataset(brief_path, repo_id, token)` pushes the `driftcall-indic-briefs` dataset (DESIGN.md §11.4).

The token is forwarded via `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` environment variables — never via argv (avoids shell-history leak). All four return a frozen `DeploymentResult` containing `repo_id`, `repo_type`, `command` (argv tuple), `return_code`, `stdout`, `stderr`, and `success`. Tests mock `subprocess.run` and `huggingface_hub.HfApi` so no network calls are issued.
