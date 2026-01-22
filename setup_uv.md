curl -Ls https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv nanogpt_env
source nanogpt_env/bin/activate
uv pip install torch numpy transformers datasets tiktoken wandb tqdm

srun --time=0:29:59 --partition normal -A a127 --environment=/users/theimer/.edf/new_axolotl.toml --pty bash