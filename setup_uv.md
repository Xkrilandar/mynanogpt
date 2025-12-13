curl -Ls https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv nanogpt_env
source nanogpt_env/bin/activate
uv pip install torch numpy transformers datasets tiktoken wandb tqdm