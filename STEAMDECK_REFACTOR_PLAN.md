# ComfyUI Steam Deck Refactoring Plan

This document outlines the comprehensive plan to refactor the ComfyUI application into a dedicated, first-class client for the Steam Deck running SteamOS in Desktop Mode.

## 1. Dependency Resolution Strategy
System-level dependencies will be installed via pacman. All Python-specific libraries will be installed via pip into a dedicated virtual environment from a requirements.txt file.

| Dependency Type | Tool   | Packages / Purpose                                           |
|---------------|--------|--------------------------------------------------------------|
| System        | pacman | `git`, `python`, `python-pytorch-rocm` (for AMD GPU support) |
| Python        | pip    | All other Python packages (numpy, Pillow, aiohttp, etc.) managed via `requirements.txt` inside a venv. |

## 2. UI/UX Overhaul Plan (Exhaustive)

This section provides a systematic catalog of UI elements and workflows, identifying issues for the Steam Deck's 1280x800 touch/trackpad interface and proposing specific solutions.

### 2.1. Global Interface Elements

| Element | Issue | Proposed Solution |
| :--- | :--- | :--- |
| **Main Menu Bar** | A row of small text links (`Queue`, `History`, `Save`, `Load`, etc.). Difficult to press accurately with a finger. Lacks icons for quick recognition. | **Replace with a persistent, top-aligned toolbar.** This toolbar will feature large, high-contrast icons for the most common actions. All text labels will be removed from the buttons themselves, but tooltips will appear on hover (useful for trackpad users). |
| **Action Buttons** | The primary "Queue Prompt" button is well-placed, but other critical actions are hidden behind text links or hotkeys. | **Add dedicated icon buttons to the new toolbar for:** `Queue Prompt` (Play icon), `Interrupt` (Stop icon), `Save Workflow` (Disk icon), `Load Workflow` (Folder icon), `New Workflow` (Plus icon), `Settings` (Gear icon). |
| **Queue/History Panel** | The floating panels for queue and history can obscure the node graph. They are activated by small menu links. | **Integrate Queue/History into a collapsible side panel.** A new button on the main toolbar will toggle a left-hand sidebar that contains the queue and history views, preventing them from overlapping the main workspace. |
| **Settings Dialog** | The settings dialog is a standard modal window with small checkboxes and text fields. | **Redesign as a full-screen overlay.** Increase the size of all interactive elements (checkboxes, dropdowns, buttons) by at least 50%. Use larger fonts and more vertical spacing to improve readability and touch accuracy. |
| **Hotkeys** | Many functions are only accessible via keyboard hotkeys (e.g., `Ctrl+S` for Save, `Ctrl+Z` for Undo, `Ctrl+C`/`V` for copy/paste). The on-screen keyboard is too cumbersome for this. | **Ensure all hotkey functionality has a visible UI equivalent.** Save/Load will be on the main toolbar. Undo/Redo buttons will be added. A new "Edit" menu on the toolbar will contain `Copy`, `Paste`, `Delete Selected`, and `Clone` actions for nodes. |

### 2.2. Node Graph & Canvas

| Element | Issue | Proposed Solution |
| :--- | :--- | :--- |
| **Canvas Panning** | Requires holding right-click or middle-click and dragging, which is awkward with a trackpad and impossible with single-touch input. | **Implement intuitive touch gestures.** Panning will be mapped to a two-finger drag gesture anywhere on the canvas. Trackpad scrolling will also be mapped to pan the canvas vertically and horizontally (with Shift). |
| **Canvas Zooming** | Relies on the mouse scroll wheel. | **Implement pinch-to-zoom.** A two-finger pinch gesture will be the primary method for zooming in and out on the canvas. Add on-screen `[+]` and `[-]` buttons in a corner of the canvas for fine-grained zoom control with single taps. |
| **Node Dragging** | Can be imprecise. Nodes are small targets to begin dragging. | **Increase the size of the node title bar.** This provides a larger, more reliable target for initiating a drag operation. |
| **Creating Connections** | The connection points (small circles) on nodes are extremely small and difficult to target. Dragging a "noodle" from one point to another requires high precision. | **Increase the visual size of connection points by 300%.** Add a larger, invisible "hitbox" around each point, so taps/clicks in the general vicinity will register. When a connection is being dragged, potential target points will "light up" and "snap" to the connection from a greater distance. |

### 2.3. Nodes & Widgets

| Element | Issue | Proposed Solution |
| :--- | :--- | :--- |
| **General Node Size** | Nodes are compact, which is efficient for large screens but makes them cramped and hard to read/interact with on a small screen. | **Increase the default width and height of all nodes by ~25%.** Increase the base font size for node titles and widget labels. |
| **Text Input Widgets** (`STRING`, `MULTILINE`) | Standard HTML text boxes. Can be difficult to tap into to activate the on-screen keyboard. | **Increase height and font size of all text input widgets.** Add a clear, visible border that changes color when the widget is focused, providing better visual feedback. Add a "Favorites" (Star) icon next to prompt boxes that opens a list of saved prompts. |
| **Number Input Widgets** (`INT`, `FLOAT`) | Small text boxes for direct number entry. Sliders are often too sensitive. | **Redesign number inputs.** The widget will show the current number. Tapping it will open a dedicated number input overlay with a large number pad and `+`/`-` buttons for incremental changes, avoiding the need for the full on-screen keyboard. |
| **Seed Widget** | Just a number input. Randomizing requires clicking a separate button. | **Combine into a single, larger widget.** Show the seed number with large `+` and `-` buttons on either side. Add a large, clear "Randomize" (Dice icon) button directly within the widget area. |
| **Toggle/Checkbox Widgets** (`BOOLEAN`) | Standard small HTML checkboxes. | **Replace with larger, more modern "toggle switches."** These are wider, easier to tap, and provide a clearer visual indication of their on/off state. |
| **Dropdown/Combo Widgets** (`COMBO`) | Standard HTML dropdown menus. The list of options is often long and uses a small font. | **Style the dropdown to open a full-width, touch-friendly list.** Each item in the list will have increased font size and vertical padding. The list will be scrollable with touch. |
| **Image/Preview Widgets** | Previews are small. No easy way to view the image in more detail. | **Add a "Maximize" icon to the corner of every image preview.** Tapping this will open the image in a full-screen, pannable, and zoomable overlay view. |

### 2.4. Menus

| Element | Issue | Proposed Solution |
| :--- | :--- | :--- |
| **Right-Click Context Menu** | Very dense, small text, requires high precision. Sub-menus are difficult to navigate with touch/trackpad. | **Completely redesign the context menu for touch.** Increase font size and add significant vertical padding (~50-75% more) to all menu items. Replace sub-menus with a two-pane system where possible (e.g., selecting "Add Node" shows categories on the left and nodes on the right within the same menu). Use icons next to common actions. |

## 3. Platform Simplification Plan

This section details the removal of code, features, and settings that are not relevant to the Steam Deck, in order to create a leaner, more focused application.

### Files/Folders to be Removed:
- **`.ci/`**: All CI scripts related to Windows builds (`update_windows/`, `windows_base_files/`, etc.) will be deleted.
- **`.github/workflows/`**: Workflows related to Windows releases (`windows_release_package.yml`, etc.) will be removed.
- **All `.bat` files**: All Windows batch scripts will be deleted (e.g., `run_cpu.bat`, `run_nvidia_gpu.bat`).
- **`extra_model_paths.yaml.example`**: This will be replaced by a hardcoded, sensible default path structure.

### Code and Features to be Removed/Modified:

- **`comfy/model_management.py`**:
    - **Remove:** All code related to NVIDIA (CUDA), Apple (MPS), Intel (XPU, oneAPI), and Windows (DirectML). This includes functions like `is_nvidia()`, `mac_version()`, and conditional imports for `torch_directml`.
    - **Simplify:** The logic for AMD GPUs (`is_amd()`) will be kept but simplified, assuming an RDNA 2 architecture.
    - **Hardcode:** Floating-point precision logic (`should_use_fp16`, etc.) will be simplified to use the optimal settings for the Steam Deck APU (likely a mix of FP32 and FP16). VRAM management states (`VRAMState`) will be simplified to a single, optimized mode.

- **`comfy/cli_args.py`**:
    - **Remove:** All command-line arguments related to non-AMD hardware will be removed from the parser. This includes `--cuda-device`, `--directml`, `--oneapi-device-selector`, etc.
    - **Remove:** Arguments for features that will be hardcoded, such as VRAM modes (`--lowvram`, `--highvram`) and precision (`--fp16-unet`).
    - **Keep:** Essential arguments like `--port`, `--auto-launch`, and `--listen` will be kept, but the code using them will be simplified.

- **`main.py`**:
    - **Remove:** All conditional logic for `os.name == 'nt'` (Windows).
    - **Remove:** Environment variable setting for CUDA, HIP (beyond a single device), and oneAPI.
    - **Simplify:** The `--auto-launch` logic will be stripped of its Windows-specific code path.

- **`folder_paths.py`**:
    - **Modify:** The default `base_path` will be changed from the application directory to a user-centric path like `~/.local/share/ComfyUI/`. The `models`, `input`, and `output` directories will be created there. This provides a cleaner installation that respects Linux file system conventions.

- **Server Configuration (`server.py`)**:
    - The default listen address is already `127.0.0.1`, which is secure and correct. No changes are needed here. The code will be implicitly simplified by the removal of command-line arguments that are no longer needed.

## 4. Proposed Installation/Setup Script
#!/bin/bash

# setup_steamdeck.sh
# Installation and setup script for ComfyUI on Steam Deck

echo "--- ComfyUI for Steam Deck Setup ---"

# The application and all its data will be installed here.
INSTALL_DIR="$HOME/ComfyUI-SteamDeck"
VENV_DIR="$INSTALL_DIR/venv"

# Ensure the script is not being run from the directory it's about to create
if [ -d "$INSTALL_DIR" ]; then
    echo "Error: Installation directory $INSTALL_DIR already exists."
    echo "Please remove it and run this script again."
    exit 1
fi

# Step 1: Install core system dependencies using pacman
echo "[1/5] Initializing pacman and installing system dependencies..."
sudo pacman-key --init
sudo pacman-key --populate archlinux
sudo pacman -S --needed --noconfirm git python python-pytorch-rocm

# Step 2: Clone the application repository
echo "[2/5] Cloning application repository to $INSTALL_DIR..."
git clone https://github.com/1456319/ComfyUI-SteamDeck.git "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Step 3: Create and activate Python virtual environment
echo "[3/5] Creating Python virtual environment in $VENV_DIR..."
python -m venv "$VENV_DIR"

# Step 4: Install Python dependencies using pip
echo "[4/5] Installing Python dependencies into the virtual environment..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# Step 5: Final user instructions
echo "[5/5] Setup complete!"
echo ""
echo "To run the application, navigate to the directory:"
echo "  cd $INSTALL_DIR"
echo "And execute the 'run_steamdeck.sh' script."
echo "Your models, inputs, and outputs should be placed according to the app's new defaults."
