# ARE_Compile
A compiler extension for NWScript projects, used to accelerate [Arelith](https://nwnarelith.com) development workflows.

### Main Features
- Parses all NWScript (.nss) files in the provided script directory and analyses the include hierarchy.
- Remembers file hashes of successfully compiled scripts, ensuring changes can be tracked.
- Maps changed files to their dependencies, intelligently compiling only files affected by the modifications, rather than the entire code base.
  - Runs on all CPU cores, just like the native [nwn_script_comp](https://github.com/niv/neverwinter.nim) binary.
  - Successfully compiled script files are moved to the given output directory.
    - An additional `secondary_output_dir` can be configured, if needed. This is useful for live testing code changes on a running NWN server instance, if set to the server’s `development` folder.

### Requirements
- ARE_Compile supports Python version 3.12 or newer. Earlier versions may work, but are untested.
- Requires [`nwn`](https://pypi.org/project/nwn/) (from PyPI) to read game files and compile scripts. Version `0.0.12` is currently recommended.
- If installed, `tqdm` will be used to display progress bars. ARE_Compile works just fine without it, though.

### Setup
- Download the repository and copy your custom NWScript files into the `input_nss` folder (or modify the `Compiler` init arguments accordingly)
- Install the `nwn` package in your Python environment (via `pip install nwn==0.0.12`).
- If the built-in autodetection of your game's installation directory fails, edit `compile.py` and pass your `nwn_install_dir` to the `Compiler` directly.

### Usage
| Command                            | Function                                                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `compile.py`                       | Uses `script_index.json` to find modified scripts and compile all scripts that are affected by the changes.                |
| `compile.py script_name`           | Compiles the script with the specified name. If an include file is given, compiles all scripts that include it.            |
| `compile.py nw_s0_*`               | Uses the given wildcard expression to find matches in the script directory. Compiles all scripts that are related to them. |
| `compile.py all` or `compile.py *` | Compile all scripts in the script directory. Required to initialise `script_index.json`.                                   |
