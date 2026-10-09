## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) to manage its Python version, virtual environment, and dependencies.

### 1. Clone the repository

Because CascadeTorch is included as a Git submodule, clone the project with:

```bash
git clone --recurse-submodules https://github.com/nikhilyadav1397/zebrafish_state_space.git
cd zebrafish_state_space
```


```bash
git clone https://github.com/PTRRupprecht/CascadeTorch.git external/CascadeTorch
```
If the repository was cloned without its submodules, initialize them afterward:

```bash
git submodule update --init --recursive
```

### 2. Install `uv`

Follow the official installation instructions at:

https://docs.astral.sh/uv/getting-started/installation/

Confirm that it is available:

```bash
uv --version
```

### 3. Create the environment and install dependencies

From the project root, run:

```bash
uv sync --frozen
```

This command:

- creates a virtual environment in `.venv/`;
- installs the Python version required by the project;
- installs the exact dependency versions recorded in `uv.lock`;
- installs CascadeTorch from `external/CascadeTorch`.

```bash
mkdir data
```

Put data into this directory.

### 3. Download appropriate CASCADE models

```bash
uv run python 0_download_model.py
```



