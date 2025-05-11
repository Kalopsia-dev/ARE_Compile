import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from glob import glob
from hashlib import sha256
from typing import Dict, Iterable, Set, Tuple

from nwn.key import Reader as KeyReader
from nwn.nwscript.comp import CompilationError
from nwn.nwscript.comp import Compiler as ScriptComp


class ScriptIndex:
    """
    An index that stores references to unique Script objects and resolves their dependencies.
    """

    # Default location for the script index file.
    PATH = os.path.join(os.path.split(__file__)[0], "tmp", "script_index.json")

    # Regular expressions for parsing script files. Pre-compiled for best performance.
    regex_comments = re.compile(r"/\*([\s\S]*?)?\*/", re.MULTILINE)
    regex_includes = re.compile(r'^#include\s+"([^"]+)"(?![^"\n]*//)', re.MULTILINE)
    regex_main = re.compile(r"^\s*void\s+main\s*\(\s*\)", re.MULTILINE)
    regex_sc = re.compile(r"^\s*int\s+StartingConditional\s*\(\s*\)", re.MULTILINE)

    def __init__(self, directory: str) -> None:
        """
        Initialises a script index object and its children Scripts using a file name.

        Args:
            directory (str): The directory containing the custom script files to compile.
        """
        print("Indexing scripts...")
        self.directory = directory
        self.scripts = dict()

        # Ensure the script index parent directory exists.
        os.makedirs(os.path.split(ScriptIndex.PATH)[0], exist_ok=True)

        # Parse all scripts in the script directory, generating a script object for each.
        {Script(file, self) for file in os.listdir(self.directory)}

        # Flattens all includes in the scripts storage, mapping each script to its dependencies.
        self.scripts = ScriptIndex.flatten_includes(self.scripts)

        # Inverts the script index so that each include file points to its children.
        self.includes = ScriptIndex.invert_index(self.scripts)

    def get_modified_scripts(self) -> Set["Script"]:
        """
        Finds all scripts that have been modified since the last compile.

        Returns:
            Set[Script]: A set of all modified scripts.
        Raises:
            FileNotFoundError: If the hash index file does not exist.
        """
        script_hashes = ScriptIndex.read_hash_index()
        return {
            script
            for script in self.scripts.values()
            if script.contents
            and (
                script.name not in script_hashes
                or script_hashes[script.name] != script.hash
            )
        }

    @staticmethod
    def flatten_includes(
        index: Dict[str, "Script"],
        prior: Dict[str, "Script"] | None = None,
        *,
        depth: int = 0,
    ) -> Dict[str, "Script"]:
        """
        Recursion that flattens the include file hierarchy up to the maximum include file depth.

        Args:
            index (Dict[str, Script]): The script index to flatten. Maps normalised script names to Script.
            prior (Dict[str, Script] | None): The previous state of the index. Used to detect changes.
            depth (int): The current recursion depth. Limited to 10 to prevent infinite loops in case of circular dependencies.

        Returns:
            Dict[str, Script]: The flattened script index. Maps normalised script names to Script.
        """
        # Combine every entry in the script index with its corresponding include's includes.
        if depth < 10:
            [
                script.includes.update(include.includes)
                for script in index.values()
                for include in set(script.includes)
            ]
            # Keep doing this until nothing changes.
            if index == prior:
                return index
            else:
                return ScriptIndex.flatten_includes(
                    index, prior=index.copy(), depth=depth + 1
                )
        else:
            print(
                "Warning: Maximum include depth exceeded. Check for circular dependencies!"
            )
            return index

    @staticmethod
    def invert_index(index: Dict[str, "Script"]) -> Dict["Script", Set["Script"]]:
        """
        Returns an inverted script index where include file point at their children files.

        Args:
            index (Dict[str, Script]): The script index to invert. Maps normalised script names to Script.

        Returns:
            Dict[Script, Set[Script]]: The inverted script index. Maps include files to their child scripts.
        """
        inverted = defaultdict(set)
        [
            inverted[include].add(script)
            for script in index.values()
            for include in script.includes
        ]

        # Validate the inverted index: if an include file has a main function, its children are also considered to have one.
        for include, scripts in dict(inverted).items():
            if include.has_main:
                if include in inverted:
                    del inverted[include]
                for script in scripts:
                    if not script.has_main:
                        script.has_main = True
            else:
                include.is_include = True
        return inverted

    def generate_hash_index(self) -> Dict[str, int]:
        """
        Returns a dictionary index of script hashes.

        This is used to check if a script has been modified since the last compile.

        Returns:
            Dict[str, int]: The index of script hashes. Maps script file names to their hashes.
        """
        # Generate a hash index of all scripts in the script index.
        hashes = {
            script.name: script.hash
            for script in self.scripts.values()
            if script.contents
        }
        return hashes

    @staticmethod
    def update_hash_index(modified: Set["Script"]) -> Dict[str, int]:
        """
        Updates the index of script hashes with the given set of modified scripts.

        Args:
            modified (Set[Script]): The set of modified scripts to update the index with.

        Returns:
            Dict[str, int]: The updated index of script hashes. Maps script file names to their hashes.
        Raises:
            FileNotFoundError: If the hash index file does not exist.
        """
        hashes = ScriptIndex.read_hash_index()
        hashes.update(
            {script.name: script.hash for script in modified if script.contents}
        )
        return hashes

    @staticmethod
    def read_hash_index() -> Dict[str, int]:
        """
        Loads the index of script hashes and returns them as a dictionary.

        Returns:
            Dict[str, int]: The index of script hashes. Maps script file names to their hashes.
        Raises:
            FileNotFoundError: If the hash index file does not exist.
        """
        try:
            with open(ScriptIndex.PATH, "r") as hashes:
                return json.load(hashes)
        except FileNotFoundError:
            return dict()

    @staticmethod
    def write_hash_index(script_hashes: Dict[str, int]) -> None:
        """
        Writes the given index of script hashes to a default location.

        Args:
            script_hashes (Dict[str, int]): The index of script hashes to write.
        """
        try:
            with open(os.path.expanduser(ScriptIndex.PATH), "w") as outfile:
                outfile.write(json.dumps(script_hashes))
        except Exception as e:
            print(f"Error: Unable to store script hashes ({e})")

    def delete_hash_index() -> None:
        """
        Deletes the index of script hashes.
        """
        try:
            os.remove(ScriptIndex.PATH)
        except OSError as e:
            print(f"Error: Unable to delete script hashes ({e})")


class Script:
    """
    A data class for script files that stores an individual script's name, path, hash, contents, and include relations.
    """

    # File suffixes for script files.
    NSS = ".nss"  # Source script file.
    NCS = ".ncs"  # Compiled script file.

    def __init__(self, name: str, script_index: ScriptIndex):
        """
        Initialises a script object and its children Scripts using a file name.

        Args:
            name (str): The name of the script file, e.g. `nw_s0_sleep`.
            script_index (ScriptIndex): The script index to use for resolving dependencies.
        """
        # Initialise the script name and path, then check if the file exists.
        self.name = Script.normalise_script_name(name)
        self.path = os.path.join(script_index.directory, f"{self.name}{Script.NSS}")
        self.contents = None

        # Initialise class variables.
        self.is_include: bool = False
        self.has_main: bool = False
        self.includes: set = set()
        self.hash: int = 0

        # Add this script to the script index if it's not included yet.
        if self.name not in script_index.scripts:
            script_index.scripts[self.name] = self

            # We can stop early if this is a base game script.
            if not os.path.isfile(self.path):
                return

            # As this script points at an existing file, parse its includes recursively.
            with open(self.path, "rb") as file:
                # First generate a hash for later use.
                self.contents = file.read()
                self.hash = int(sha256(self.contents).hexdigest(), 16)
                # Afterwards, decode the file, strip block comments and analyse the remaining file contents.
                contents = re.sub(
                    script_index.regex_comments,
                    "",
                    self.contents.decode(encoding="ISO-8859-1"),
                )
                self.has_main = re.search(
                    script_index.regex_main, contents
                ) or re.search(script_index.regex_sc, contents)
                self.includes = {
                    Script(include, script_index)
                    if include not in script_index.scripts
                    else script_index.scripts[include]
                    for include in re.findall(script_index.regex_includes, contents)
                }

    def __hash__(self) -> int:
        """
        Returns this script's previously calculated hash value.
        This is used to compare scripts and check if thir contents have changed.

        Returns:
            int: The hash value of the script.
        """
        return self.hash

    def __eq__(self, other) -> bool:
        """
        Compares scripts based on their hash values.

        Args:
            other (Script): The script to compare with.

        Returns:
            bool: True if the scripts are equal, False otherwise.
        """
        if not isinstance(other, Script):
            return False
        return self.hash == other.hash and self.path == other.path

    def __repr__(self) -> str:
        """
        Returns a string representation of the script.
        """
        return f"{self.name} <{'Script' if not self.is_include else 'Include'} with {len(self.includes)} include(s)>"

    @staticmethod
    def normalise_script_name(script_name: str):
        """
        Normalises a script name by removing the file extension and converting it to lowercase.

        Args:
            script_name (str): The name of the script to normalise.

        Returns:
            str: The normalised script name.
        """
        # Remove the file extension and convert to lowercase.
        return os.path.splitext(script_name)[0].lower()


class Compiler:
    """
    A compiler that intelligently compiles scripts based on their include relations and hash changes.
    """

    def __init__(
        self,
        params: list,
        script_dir: str,
        output_dir: str,
        nwn_install_dir: str,
        secondary_output_dir=None,
        num_workers: int = -1,
        **_,
    ) -> None:
        """
        Initialises a compiler object and compiles the given script files.

        Args:
            params (list): The command-line parameters to pass to the compiler, e.g. `["all"]` or `["nw_s0_sleep"]`.
            script_dir (str): The directory containing the custom script files to compile. Parsed by `ScriptIndex`.
            output_dir (str): The directory to write the compiled script files to.
            nwn_install_dir (str): The directory containing the NWN installation files.
            secondary_output_dir (str, optional): An optional additional output directory for compiled scripts.
            num_workers (int, optional): The number of worker threads to use for compilation. Defaults to the number of CPU cores.
        """
        # Check whether the provided paths exist. If output directories are missing, create them.
        if not os.path.exists(script_dir):
            print(f'Error: Unable to locate script directory at "{script_dir}"')
            exit(1)
        if not os.path.exists(nwn_install_dir):
            print(f'Error: Unable to locate NWN installation at "{nwn_install_dir}"')
            exit(1)
        key_file = os.path.join(nwn_install_dir, "data", "nwn_base.key")
        if not os.path.isfile(key_file):
            print(f'Error: Unable to locate NWN key file at "{key_file}"')
            exit(1)
        if secondary_output_dir:
            os.makedirs(secondary_output_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Store the current time to calculate the total execution time later.
        self.start_time = time.time()

        # Store the given parameters and directories.
        self.nwn_install_dir = nwn_install_dir
        self.output_dirs = [output_dir]
        if secondary_output_dir:
            self.output_dirs.append(secondary_output_dir)

        # Determine the number of workers to use for compilation.
        self.num_workers = num_workers if num_workers != -1 else os.cpu_count() or 1

        # Compile all scripts if the output directory is empty or there is no hash index.
        output_files = len(glob(os.path.join(output_dir, f"*{Script.NCS}")))
        script_hashes = ScriptIndex.read_hash_index()
        if params != ["all"] and not (output_files and script_hashes):
            print("All scripts will be compiled to initialise the index.", end="\n\n")
            params = ["all"]

        # Generate a script index to analyse the include structure.
        self.script_index = ScriptIndex(script_dir)

        # Load the NWN key file. It contains the base scripts.
        self.key_reader = KeyReader(key_file)

        # Determine the compile mode based on the given parameters and availability of a hash index.
        if params:
            # If there are any arguments, use them to determine what to compile.
            param = Script.normalise_script_name(params[0])
            if param == "all":
                self.compile_all()
            elif param[-1] == "*":
                self.compile_wildcard(param)
            else:
                self.compile_script(param)
        else:
            # Fallback to the default behaviour: compile all modified scripts.
            self.compile_modified()

    def compile(self, scripts: Script | Iterable[Script]) -> bool:
        """
        Compiles a given script file or iterable of scripts, then writes the results to the output directory.

        Args:
            scripts (Script | Iterable[Script]): The script or iterable of scripts to compile.

        Returns:
            bool: True if the compilation of all scripts was successful, False otherwise.
        """
        # First, normalise the input to a set of scripts.
        if isinstance(scripts, Script):
            scripts = {scripts}

        # Prepare the thread pool executor for parallel compilation.
        thread_data = threading.local()
        cancelled = False

        def compiler() -> ScriptComp:
            """
            Returns a thread-local `ScriptComp` instance.
            """
            if not hasattr(thread_data, "compiler"):
                # single shared KeyReader is fine, load_script_contents is thread-safe
                thread_data.compiler = ScriptComp(
                    resolver=self.load_script_contents,
                    debug_info=False,
                    max_include_depth=64,
                )
            return thread_data.compiler

        def compile_in_thread(script: Script) -> Tuple[str, bytes | None]:
            """
            Compiles a script in a separate thread and returns the result.

            Args:
                script (Script): The script to compile.

            Returns:
                Tuple[str, bytes]: The name of the compiled script and its binary contents.
            Raises:
                CompilationError: If the compilation fails.
            """
            print(f"Compiling: {script.name}{Script.NSS}")
            try:
                # Attempt to compile the script using the thread-local compiler instance.
                ncs_bytes = compiler().compile(script.name)[0]
            except CompilationError as error:
                # Convert compile errors into the desired output format.
                error_msg = error.message.splitlines()[0].split(" [")[0]
                if error_msg.endswith("ERROR: NO FUNCTION MAIN IN SCRIPT"):
                    # Trying to compile an include file is fine, no need to raise.
                    return f"{script.name}{Script.NCS}", None
                # File not found errors already get displayed by load_script_contents.
                if not error_msg.endswith("ERROR: FILE NOT FOUND"):
                    # All other errors are printed.
                    print(f"\n{error_msg}")
                raise error
            return f"{script.name}{Script.NCS}", ncs_bytes

        try:
            # Asynchronously compile all scripts in the given set.
            # We can use threads because ScriptComp is not subject to the GIL.
            with ThreadPoolExecutor(
                max_workers=min(self.num_workers, len(scripts))
            ) as executor:
                compiled = executor.map(
                    compile_in_thread, sorted(scripts, key=lambda s: s.name)
                )
                # Unpack the results.
                compiled = {
                    ncs_name: ncs for ncs_name, ncs in compiled if ncs is not None
                }
        except CompilationError:
            # Compilation errors should stop all threads.
            print(
                "\nStopping processing on first error.\n\n1 error; see above for context.",
                end="\n\n",
            )
            cancelled = True
        except KeyboardInterrupt:
            # Gracefully handle keyboard interrupts.
            print("\nStopping processing on user request.", end="\n\n")
            cancelled = True
        except Exception as e:
            # Handle all other exceptions.
            print(e.with_traceback())
            cancelled = True
        finally:
            if cancelled:
                executor.shutdown(wait=False, cancel_futures=True)
                print("Processing aborted.", end="\n\n")
                return False

        # If we got here, all scripts compiled successfully. Copy their binaries to the output dirs.
        print("\nWriting script(s) to output folder...")
        for ncs_name, ncs in compiled.items():
            for output_dir in self.output_dirs:
                # Write the compiled script to the output directory.
                with open(os.path.join(output_dir, ncs_name), "wb") as outfile:
                    outfile.write(ncs)
        print(
            f"Success!\n\nTotal Execution time = {time.time() - self.start_time:.4f} seconds",
            end="\n\n",
        )
        return True

    @lru_cache(maxsize=None)
    def load_script_contents(self, script_name: str) -> bytes | None:
        """
        Loads a script from the script index or the NWN installation directory.

        Args:
            script_name (str): The file name of the script to load, e.g. `nw_s0_sleep.nss`.

        Returns:
            bytes | None: The contents of the script file, or None if the file could not be found.
        """
        # First, check if the script is in the script index.
        script_name = Script.normalise_script_name(script_name)
        if script_name in self.script_index.scripts:
            script = self.script_index.scripts[script_name]
            if script.contents:
                # If yes, directly return its contents.
                return script.contents
        # If not, we might find it in the NWN installation directory.
        # First, check the overrides, which take precedence over key files.
        script_name += Script.NSS
        override = os.path.join(self.nwn_install_dir, "ovr", script_name)
        if os.path.isfile(override):
            with open(override, "rb") as file:
                return file.read()
        try:
            # If it's not here, attempt to load it from the NWN key file.
            return self.key_reader.read_file(script_name)
        except FileNotFoundError:
            # We have no options left. Print an error message and return None.
            print(f"\n{script_name}: ERROR: FILE NOT FOUND")
            return None

    def find_related_scripts(self, scripts: Set[Script]) -> Set[Script]:
        """
        Returns a set of all scripts (except includes) affected by changes to the given set of scripts.

        Args:
            scripts (Set[Script]): The set of scripts to check for dependencies.

        Returns:
            Set[Script]: A set of all scripts affected by changes to the provided scripts.
        """
        # Handle single script inputs.
        if isinstance(scripts, Script):
            scripts = {scripts}
        # For include files, add all scripts affected by the change to the set. Otherwise, just add the script.
        related = set()
        for script in scripts:
            if script not in self.script_index.includes:
                # If the script is not an include file, add it to the set.
                related.add(script)
            else:
                # Otherwise, add all scripts affected by the change to the set.
                related.update(self.script_index.includes[script])
        # Finally, remove all include files from the set and return it.
        return related - set(self.script_index.includes.keys())

    def compile_script(self, script_name: str) -> None:
        """
        Compiles an individual script file.

        Args:
            script_name (str): The name of the script to compile, e.g. `nw_s0_sleep`.
        """
        script = (
            self.script_index.scripts[script_name]
            if script_name in self.script_index.scripts
            else None
        )
        if script and script.contents:
            # Check if this is an include file.
            if not script.is_include:
                # It is not. Generate the given script's hash. We'll remember it if this compile is successful.
                script_hashes = ScriptIndex.update_hash_index({script})
                # Compile the file. If the operation is successful, update the hash index.
                if self.compile(script):
                    ScriptIndex.write_hash_index(script_hashes)
            else:
                # It is an include file. Compile all dependencies.
                print("Include file detected. Checking dependencies...", end="\n\n")
                to_compile = self.find_related_scripts(script)
                if to_compile:
                    print(
                        f"{len(to_compile)} script(s) include {script.name}.",
                        end="\n\n",
                    )
                    self.compile_set(to_compile, script)
                else:
                    print(f"No scripts include {script.name}.")
        else:
            print(f"Error: Unable to find {script_name}{Script.NSS}")

    def compile_all(self) -> None:
        """
        Compiles all scripts in the main ScriptIndex directory and copies the results to the output directory.

        This is used to initialise the script index.
        """
        # Clear all files within the output directory, as well as the script index.
        self.reset_output_directory()
        ScriptIndex.delete_hash_index()
        # Load the hashes of all scripts in the script directory. We'll remember them if this compile is successful.
        script_hashes = self.script_index.generate_hash_index()
        # Now batch compile all script files with compile. If successful, update the hash index.
        scripts = {
            script
            for script in self.script_index.scripts.values()
            if script.contents and not script.is_include
        }
        if self.compile(scripts):
            ScriptIndex.write_hash_index(script_hashes)

    def compile_set(
        self,
        scripts: Script | Set[Script],
        modified: Script | Set[Script],
    ) -> None:
        """
        Compiles all scripts in a given set within the temporary folder and copies the results to the output directory, if successful.

        Args:
            scripts (Script | Set[Script]): The set of scripts to compile.
            modified (Script | Set[Script]): The set of modified scripts. Used to update the hash index.
        """
        # Handle single script inputs.
        if isinstance(scripts, Script):
            scripts = {scripts}
        if isinstance(modified, Script):
            modified = {modified}
        # Load the script hashes of all scripts in the given set. We'll remember them if this compile is successful.
        script_hashes = ScriptIndex.update_hash_index(modified)
        # Check if the operation was successful.
        if self.compile(scripts):
            ScriptIndex.write_hash_index(script_hashes)

    def compile_modified(self) -> None:
        """
        Compile all modified and affected scripts, as identified by the ScriptIndex.
        """
        # Generate a set of new and modified scripts and check what needs to be compiled.
        modified = self.script_index.get_modified_scripts()
        to_compile = self.find_related_scripts(modified)
        if modified and to_compile:
            print(
                f"{len(modified)} change(s) found. {len(to_compile)} affected script(s) will be compiled.",
                end="\n\n",
            )
            self.compile_set(to_compile, modified)
        else:
            print("All scripts are up to date.")

    def compile_wildcard(self, script_name: str) -> None:
        """
        Compiles all scripts matching the given wildcard.

        Args:
            script_name (str): The wildcard to match against script names.
        """
        # Compile all scripts if a blank wildcard is given.
        if script_name == "*":
            return self.compile_all()
        # Check if the wildcard matches any scripts.
        regex = re.compile(script_name.replace("*", ".*"), re.IGNORECASE)
        matches = {
            script
            for script in self.script_index.scripts.values()
            if script.contents and regex.match(script.name)
        }
        to_compile = self.find_related_scripts(matches)
        if matches and to_compile:
            # Print a list of all matches, then compile the affected scripts.
            if len(matches) < 30:
                print(f"\n{len(matches)} match(es) found:")
                [
                    print(f"- {script_name}{Script.NSS}")
                    for script_name in sorted([script.name for script in matches])
                ]
            else:
                print(f"\n{len(matches)} match(es) found.")
            print(
                f"\n{len(to_compile)} related script(s) will be compiled.", end="\n\n"
            )
            self.compile_set(to_compile, matches)
        else:
            print("No matches found.")

    def reset_output_directory(self) -> None:
        """
        Removes all compiled script files from the output directories.
        """
        all(
            not os.remove(file)
            for directory in self.output_dirs
            for file in glob(os.path.join(directory, f"*{Script.NCS}"))
        )


if __name__ == "__main__":
    # Create a script index and compile based on the given command-line parameters.
    Compiler(
        script_dir=os.path.join(os.getcwd(), "scripts"),
        output_dir=os.path.join(os.getcwd(), "compiled-scripts"),
        nwn_install_dir=None,  # TODO: Add your NWN install directory here!
        params=sys.argv[1:],
    )
