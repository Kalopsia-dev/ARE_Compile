# ARE_Compile
Compiler extensions used in development workflows on the [Arelith persistent role playing server](https://nwnarelith.com).

### Main Features
- Parse all NWScript (.nss) files in the provided `script_dir` and analyses their include file hierarchy.
- Keep track of scripts that `nwn_script_comp` successfully compiled. Stores this information in a `script_index.json` file with persistent file hashes.
- Use the above information to map modified files to all their dependencies and intelligently compile only these affected files, rather than the entire code base.
  - Runs on all CPU cores, just like the native [nwn_script_comp](https://github.com/niv/neverwinter.nim) binary.
  - Successfully compiled script files are moved to the given `output_dir`.
    - A `secondary_output_dir` can be configured, if needed. This is useful for live testing code changes on a running NWN server instance, if set to the server’s `development` folder.

### Environment Setup
- By default, ARE_Compile looks for a `nwn_script_comp` binary in the same folder as this Python script.
- The default input directory is called `scripts` and should be in the same folder - just like `compiled-scripts`, the output directory.
- Presently, there is no built-in autodetection of the NWN install directory, so `compile.py` must be updated to pass a valid `nwn_install_dir` to the `Compiler` object on creation.

### Usage
| Command                            | Function                                                                                                                   |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `compile.py`                       | Uses `script_index.json` to find modified scripts and compile all scripts that are affected by the changes.                |
| `compile.py script_name`           | Compiles the script with the specified name. If an include file is given, compiles all scripts that include it.            |
| `compile.py nw_s0_*`               | Uses the given wildcard expression to find matches in the script directory. Compiles all scripts that are related to them. |
| `compile.py all` or `compile.py *` | Compile all scripts in the script directory. Required to initialise `script_index.json`.                                   |

### Requirements
- ARE_Compile is known to be compatible with Python version 3.12 or newer. Earlier versions may or may not work as expected.
- Requires the [`nwn`](https://pypi.org/project/nwn/) Python package to read game files and compile scripts.
  - ARE_Compile has been developed with version `0.0.12`. Installing this version is recommended, as `nwn.py` is still in early development, so its API may change.
