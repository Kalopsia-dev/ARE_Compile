# ARE_Compile
A Python-based wrapper for [nwn_script_comp](https://github.com/niv/neverwinter.nim), used in development workflows on the [Arelith persistent role playing server](https://nwnarelith.com).

### Main Features
- Parses all NWScript (.nss) files in the provided `script_dir` and analyses their include file hierarchy.
- Keeps track of scripts that `nwn_script_comp` successfully compiled. Stores this information in a `script_index.json` file with persistent file hashes.
- Uses the above information to map modified files to all their dependencies and intelligently compile only these affected files, rather than the entire code base.
- Successfully compiled script files are moved to the given `output_dir`.
  - A `secondary_output_dir` can be configured, if needed. This is useful for live testing code changes on a running NWN server instance, if set to the server’s `development` folder.

### Environment Setup
- By default, ARE_Compile looks for a `nwn_script_comp` binary in the same folder as this Python script.
- The default input directory is called `scripts` and should be in the same folder - just like `compiled-scripts`, the output directory.
- If `nwn_script_comp`’s built-in autodetection of NWN root and user directory is unreliable, specific folders can be selected via the `nwn_install_dir` and `nwn_user_dir` parameters of the Compiler class.

### Usage
| Command                            | Function                                                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `compile.py`                       | Uses `script_index.json` to find modified scripts and compile all scripts that are affected by the changes.                |
| `compile.py script_name`           | Compiles the script with the specified name. If an include file is given, compiles all scripts that include it.            |
| `compile.py nw_s0_*`               | Uses the given wildcard expression to find matches in the script directory. Compiles all scripts that are related to them. |
| `compile.py all` or `compile.py *` | Compile all scripts in the script directory. Required to initialise `script_index.json`.                                   |

### Requirements
- ARE_Compile is compatible with Python version 3.11 or newer. No additional Python dependencies are required.
- Uses version [1.7.0](https://github.com/niv/neverwinter.nim/releases/tag/1.7.0) or newer of `nwn_script_comp` to compile scripts.
