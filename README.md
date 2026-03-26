# NeuroAgent

NeuroAgent is a Windows desktop BCI automation prototype that combines:
- c-VEP EEG decoding (EEG2Code-style) for user selection
- OmniParser for UI element detection on the active window
- Ollama + Gemma 3 for action proposal and goal verification

The app shows selectable actions on a secondary display and executes the selected desktop action.

## Current status

This repository currently contains:
- Core NeuroAgent app code
- Preloaded BCI assets:
  - `data/VP1.mat`
  - `model/EEG2Code_model.hdf5`
- A one-time setup script: `setup.bat`

## Platform and requirements

This project is currently **Windows-focused** and expects:
- Windows 10/11
- Python 3.10 (via Conda environment `neuralink`)
- Ollama installed and available in PATH
- A second monitor connected (BCI display opens on non-primary monitor)
- OmniParser folder and model weights available

You must install these manually before running `setup.bat`:
- Ollama: https://ollama.com/download
- Miniconda or Anaconda
- CUDA drivers (optional, for GPU acceleration)
- OmniParser weights (YOLO + Florence2)

## Quick start (recommended)

1. Open Command Prompt in this project folder.
2. Run:

```bat
setup.bat
```

`setup.bat` does the following:
1. Verifies Ollama is installed.
2. Starts `ollama serve` in the background.
3. Pulls `gemma3:4b`.
4. Creates Conda env `neuralink` with Python 3.10 if missing.
5. Installs dependencies from `requirements.txt` (if present).
6. Creates `NeuroAgent.bat` launcher.

Then launch with:

```bat
NeuroAgent.bat
```

## Manual setup (if you prefer)

```bat
conda create -n neuralink python=3.10 -y
conda activate neuralink
python -m pip install --upgrade pip
```

Install core packages used by this repo:

```bat
python -m pip install requests pillow numpy scipy tensorflow pyautogui screeninfo pywin32
```

Start Ollama and pull model:

```bat
ollama serve
ollama pull gemma3:4b
```

Run the app:

```bat
python main.py
```

## OmniParser integration

At startup, NeuroAgent resolves `omniparser_dir` in this order:
1. `OMNIPARSER_DIR` environment variable
2. `<repo>/OmniParser`
3. Sibling folder `../OmniParser`

Expected files:
- `weights/icon_detect/model.pt`
- `weights/icon_caption_florence/`

If not found, startup fails with an OmniParser initialization error.

## Runtime behavior

`main.py` performs:
1. Ensure Ollama server is reachable (starts it if needed).
2. Load app config (`AppConfig`).
3. Warm-load configured model (default `gemma3:4b`).
4. Initialize BCI display and decoder.
5. Start controller loop.

Keyboard controls:
- `Space`: scan environment and refresh action options
- `Q` or `Esc`: quit

Debug artifacts are written to:
- `bci_debug/<timestamp>/...`
- `bci_debug/llm_proposed_<timestamp>.json`
- `bci_debug/agent_actions.jsonl`

## Configuration

Default settings live in `app_config.py`, including:
- `llm_model = "gemma3:4b"`
- OmniParser thresholds
- Items per page and wait timings

To override OmniParser location for one session:

```bat
set OMNIPARSER_DIR=C:\path\to\OmniParser
NeuroAgent.bat
```

## Troubleshooting

- `Ollama is not installed`  
  Install Ollama and reopen terminal so PATH updates.

- `Conda not found`  
  Install Miniconda/Anaconda and use a Conda-enabled terminal.

- `Second monitor not detected`  
  Connect/enable a second display. The BCI UI requires a non-primary monitor.

- `OmniParser folder not found` or missing weights  
  Set `OMNIPARSER_DIR` correctly and verify YOLO/Florence2 weights are present.

- `requirements.txt not found` during `setup.bat`  
  This repo currently has no `requirements.txt`. Install packages manually (see Manual setup), or add your own `requirements.txt`.

## Repository layout

- `main.py` - app entrypoint
- `controller.py` - BCI-agent control loop
- `bci_decoder.py` - EEG decoding logic
- `bci_display.py` - second-monitor BCI UI
- `omniparser_engine.py` - screen capture + OmniParser integration
- `ranking.py` - action ranking/generation
- `goal_verifier.py` - screenshot-based goal verification via Ollama vision
- `setup.bat` - one-time environment setup

