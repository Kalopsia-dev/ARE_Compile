# ARE_Compile
A Python-based wrapper for [nwn_script_comp](https://github.com/niv/neverwinter.nim), used in development workflows on the [Arelith persistent role playing server](https://nwnarelith.com).

### Main Features
- Parses all NWScript (.nss) files in the given `SCRIPT_DIR` and analyses their include file hierarchy.
- Keeps track of scripts that `nwn_script_comp` successfully compiled. Stores this information in a `script_index.json` file with persistent file hashes.
- Uses the above information to map modified files to all their dependencies and intelligently compile only these affected files, rather than the entire code base.
- Successfully compiled script files are moved to the given `OUTPUT_DIR` and, if a `--live` flag is specified in the params, the `SERVER_DIR`.

### Usage
| Command                            | Function                                                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `compile.py`                       | Uses `script_index.json` to find modified scripts and compile all scripts that are affected by the changes.                |
| `compile.py script_name`           | Compiles the script with the specified name. If an include file is specified, compiles all scripts that include it.        |
| `compile.py nw_s0_*`               | Uses the given wildcard expression to find matches in the script directory. Compiles all scripts that are related to them. |
| `compile.py all` or `compile.py *` | Compile all scripts in the script directory. Required to initialise `script_index.json`.                                   |

### Requirements
- ARE_Compile is compatible with Python version 3.11 or newer. No additional Python dependencies are required.
- Uses version [1.7.0](https://github.com/niv/neverwinter.nim/releases/tag/1.7.0) or newer of `nwn_script_comp` to compile scripts.
