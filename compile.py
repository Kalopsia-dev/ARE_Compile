import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from glob import glob
from hashlib import sha256

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
        {
            Script(script_name, script_index=self)
            for script_name in glob(os.path.join(directory, f"*{Script.NSS}"))
        }

        # Flattens all includes in the scripts storage, mapping each script to its dependencies.
        self.scripts = ScriptIndex.flatten_includes(self.scripts)

        # Inverts the script index so that each include file points to its children.
        self.includes = ScriptIndex.invert_index(self.scripts)

    def get_modified_scripts(self) -> set["Script"]:
        """
        Finds all scripts that have been modified since the last compile.

        Returns:
            set[Script]: A set of all modified scripts.
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
    def flatten_includes(index: dict[str, "Script"]) -> dict[str, "Script"]:
        """
        Recursion that flattens the include file hierarchy, mapping each script to all its dependencies.

        Args:
            index (dict[str, Script]): The script index to flatten.

        Returns:
            dict[str, Script]: The flattened script index. Maps normalised script names to Script.
        """

        def count_includes(index: dict[str, "Script"]) -> dict[str, int]:
            """
            Returns a dictionary of include counts for each script in the index.
            """
            return {
                script.name: len(script.includes)
                for script in index.values()
                if script.includes
            }

        last_count = None
        for _ in range(10):
            if last_count == (current_count := count_includes(index)):
                # If the include count hasn't changed since the last iteration, we can stop.
                break
            # Combine every entry in the script index with its corresponding include's includes.
            any(
                script.includes.update(include.includes)
                for script in index.values()
                for include in set(script.includes)
                if script.includes
                and (
                    last_count is None  # First iteration.
                    or (last_count[script.name] != current_count[script.name])
                )
            )
            # Update the stored include count for the next iteration.
            last_count = current_count
        return index

    @staticmethod
    def invert_index(index: dict[str, "Script"]) -> dict["Script", set["Script"]]:
        """
        Returns an inverted script index where include files point at their children.

        Args:
            index (dict[str, Script]): The script index to invert. Maps normalised script names to Script objects.

        Returns:
            dict[Script, set[Script]]: The inverted script index. Maps include files to their child scripts.
        """
        # Prepare a dictionary to store the inverted script index.
        includes = dict.fromkeys(
            {include for script in index.values() for include in script.includes},
        )
        # Add all scripts to the corresponding include keys.
        for script in index.values():
            for include in script.includes:
                # If we haven't seen this include file before, create a new set for it.
                if not include.is_include:
                    include.is_include = True
                    includes[include] = {script}
                else:
                    includes[include].add(script)
                # If an include file has a main function, its children inherit it.
                if include.has_main:
                    script.has_main = True
        return includes

    def generate_hash_index(self) -> dict[str, int]:
        """
        Returns a dictionary index of script hashes.

        This is used to check if a script has been modified since the last compile.

        Returns:
            dict[str, int]: The index of script hashes. Maps script file names to their hashes.
        """
        # Generate a hash index of all scripts in the script index.
        hashes = {
            script.name: script.hash
            for script in self.scripts.values()
            if script.contents
        }
        return hashes

    @staticmethod
    def update_hash_index(modified: set["Script"]) -> dict[str, int]:
        """
        Updates the index of script hashes with the given set of modified scripts.

        Args:
            modified (set[Script]): The set of modified scripts to update the index with.

        Returns:
            dict[str, int]: The updated index of script hashes. Maps script file names to their hashes.
        """
        hashes = ScriptIndex.read_hash_index()
        hashes.update(
            {script.name: script.hash for script in modified if script.contents}
        )
        return hashes

    @staticmethod
    def read_hash_index() -> dict[str, int]:
        """
        Loads the index of script hashes and returns them as a dictionary.

        Returns:
            dict[str, int]: The index of script hashes. Maps script file names to their hashes.
        """
        if os.path.exists(ScriptIndex.PATH):
            with open(ScriptIndex.PATH, "r") as hashes:
                return json.load(hashes)
        else:
            return dict()

    @staticmethod
    def write_hash_index(script_hashes: dict[str, int]) -> None:
        """
        Writes the given index of script hashes to a default location.

        Args:
            script_hashes (dict[str, int]): The index of script hashes to write.
        """
        try:
            with open(os.path.expanduser(ScriptIndex.PATH), "w") as outfile:
                outfile.write(json.dumps(script_hashes))
        except Exception as e:
            print(f"Error: Unable to store script hashes ({e})")

    def delete_hash_index() -> None:
        """
        Deletes the index of script hashes, if it exists.
        """
        if os.path.exists(ScriptIndex.PATH):
            os.remove(ScriptIndex.PATH)


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
        if self is other:
            return True
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
        script_name = os.path.basename(script_name)
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
        self.num_workers = num_workers if num_workers > 0 else max(os.cpu_count(), 1)

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

    def compile(
        self,
        scripts: Script | set[Script],
        new_script_hashes: dict[str, int] = None,
    ) -> None:
        """
        Compiles a given script file or iterable of scripts, then writes the results to the output directory.

        Args:
            scripts (Script | Iterable[Script]): The script or iterable of scripts to compile.
            new_script_hashes (dict[str, int], optional): The new hashes to store if compilation is successful.

        Returns:
            bool: True if the compilation of all scripts was successful, False otherwise.
        """
        # First, normalise the input to a set of scripts.
        if isinstance(scripts, Script):
            scripts = {scripts}

        # Prepare the thread pool executor for parallel compilation.
        locals = threading.local()

        def init_thread() -> None:
            """
            Initialises a thread-local compiler instance.
            """
            locals.compiler = ScriptComp(
                resolver=self.load_script_contents,
                debug_info=False,
                max_include_depth=64,
            )

        def compile_in_thread(script: Script) -> tuple[str | None, bytes | None]:
            """
            Compiles a script in a separate thread and returns the result.

            Args:
                script (Script): The script to compile.

            Returns:
                tuple[str | None, bytes | None]: The name of the compiled script and its binary, if successful.
            Raises:
                CompilationError: If the compilation fails.
            """
            print(f"Compiling: {script.name}{Script.NSS}")
            try:
                # Attempt to compile the script using the thread-local compiler instance.
                ncs_bytes = locals.compiler.compile(script.name)[0]
                return f"{script.name}{Script.NCS}", ncs_bytes
            except CompilationError as error:
                # Format, print and raise the error. It will then interrupt the thread pool.
                print(f"\n{error.message.splitlines()[0].split(' [')[0]}")
                raise error

        try:
            # We will use threads because ScriptComp is not subject to the GIL.
            with ThreadPoolExecutor(
                initializer=init_thread,
                max_workers=min(self.num_workers, len(scripts)),
            ) as executor:
                # Asynchronously compile all scripts in the given set.
                compiled = {
                    ncs_name: ncs_bytes
                    for ncs_name, ncs_bytes in executor.map(
                        compile_in_thread,
                        sorted(scripts, key=lambda script: script.name),
                    )
                    if ncs_bytes is not None
                }
            successful = True
        except CompilationError:
            # Compilation errors should stop all threads.
            print(
                "\nStopping processing on first error.\n\n1 error; see above for context.",
                end="\n\n",
            )
            successful = False
        except KeyboardInterrupt:
            # Gracefully handle keyboard interrupts.
            print("\nStopping processing on user request.", end="\n\n")
            successful = False
        except Exception as e:
            # Handle all other exceptions.
            print(e.with_traceback())
            successful = False
        finally:
            if not successful:
                # If anything went wrong, cancel all remaining tasks immediately.
                executor.shutdown(wait=False, cancel_futures=True)
                print("Processing aborted.", end="\n\n")
                return

        # If we got here, all scripts compiled successfully. Copy their binaries to the output dirs.
        print("\nWriting script(s) to output folder...")
        for ncs_name, ncs in compiled.items():
            for output_dir in self.output_dirs:
                # Write the compiled script to the output directory.
                with open(os.path.join(output_dir, ncs_name), "wb") as outfile:
                    outfile.write(ncs)
        if new_script_hashes:
            # If we have a new hash index, write it to the hash file now.
            ScriptIndex.write_hash_index(new_script_hashes)

        # Finally, print the total execution time.
        compile_time = time.time() - self.start_time
        print(f"Success!\n\nTotal Execution time = {compile_time:.4f} seconds\n")

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
            # We have no options left.
            return None

    def find_related_scripts(self, scripts: Script | set[Script]) -> set[Script]:
        """
        Returns a set of all scripts (except includes) affected by changes to the given set of scripts.

        Args:
            scripts (Script | set[Script]): The set of scripts to check for dependencies.

        Returns:
            set[Script]: A set of all scripts affected by changes to the provided scripts.
        """
        # Handle single script inputs.
        if isinstance(scripts, Script):
            scripts = {scripts}
        # For include files, add all scripts affected by the change to the set. Otherwise, just add the script.
        related = set()
        for script in scripts:
            if script not in self.script_index.includes:
                # If the script is not an include file, simply add it to the set.
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
        # We can only compile scripts in the script index.
        if script_name not in self.script_index.scripts:
            print(f"Error: Unable to find {script_name}{Script.NSS}")
            return
        # Exclude base game scripts.
        script = self.script_index.scripts[script_name]
        if not script.contents:
            print(f"Error: {script_name}{Script.NSS} is a base game script.")
            return
        # Next, check if the script is an include file.
        if script.is_include:
            # It is, so compile all dependencies.
            print("Include file detected. Checking dependencies...", end="\n\n")
            to_compile = {
                script
                for script in self.find_related_scripts(script)
                if script.has_main
            }
            if not to_compile:
                print(f"No scripts include {script.name}.")
                return
            print(
                f"{len(to_compile)} script(s) include {script.name}.",
                end="\n\n",
            )
            self.compile(
                scripts=to_compile,
                new_script_hashes=ScriptIndex.update_hash_index({script}),
            )
        elif script.has_main:
            # Compile the file. If the operation is successful, update the hash index.
            self.compile(
                scripts=script,
                new_script_hashes=ScriptIndex.update_hash_index({script}),
            )
        else:
            # The script has no main function, so we can't compile it.
            print(
                f"\nError: {script_name}{Script.NSS} is an unused include file with no main function.",
                end="\n\n",
            )
            return

    def compile_all(self) -> None:
        """
        Compiles all scripts in the main ScriptIndex directory and copies the results to the output directory.

        This is used to initialise the script index.
        """
        # Locate all relevant script files in the script index.
        scripts = {
            script
            for script in self.script_index.scripts.values()
            if script.contents and script.has_main
        }
        # Clear old output files, including the hash index.
        self.reset_output_directory()
        # Batch compile all script files. If successful, update the hash index with the current file hashes.
        self.compile(
            scripts=scripts, new_script_hashes=self.script_index.generate_hash_index()
        )

    def compile_modified(self) -> None:
        """
        Compile all modified and affected scripts, as identified by the ScriptIndex.
        """
        # Generate a set of new and modified scripts and check what needs to be compiled.
        modified = self.script_index.get_modified_scripts()
        to_compile = {
            script for script in self.find_related_scripts(modified) if script.has_main
        }
        if not modified or not to_compile:
            print("All scripts are up to date.")
            return
        print(
            f"{len(modified)} change(s) found. {len(to_compile)} affected script(s) will be compiled.",
            end="\n\n",
        )
        self.compile(
            scripts=to_compile,
            new_script_hashes=ScriptIndex.update_hash_index(modified),
        )

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
        # We can only compile scripts with a main function.
        to_compile = {
            script for script in self.find_related_scripts(matches) if script.has_main
        }
        if not matches or not to_compile:
            print("No matches found.")
            return
        # Print a list of all matches, then compile the affected scripts.
        if len(matches) < 30:
            print(f"\n{len(matches)} match(es) found:")
            [
                print(f"- {script_name}{Script.NSS}")
                for script_name in sorted([script.name for script in matches])
            ]
        else:
            print(f"\n{len(matches)} match(es) found.")
        print(f"\n{len(to_compile)} related script(s) will be compiled.", end="\n\n")
        self.compile(
            scripts=to_compile, new_script_hashes=ScriptIndex.update_hash_index(matches)
        )

    def reset_output_directory(self) -> None:
        """
        Removes all compiled script files from the output directories.
        """
        ScriptIndex.delete_hash_index()
        all(
            not os.remove(file)
            for directory in self.output_dirs
            for file in glob(os.path.join(directory, f"*{Script.NCS}"))
        )


if __name__ == "__main__":
    import sys

    # Create a script index and compile based on the given command-line parameters.
    Compiler(
        script_dir=os.path.join(os.getcwd(), "scripts"),
        output_dir=os.path.join(os.getcwd(), "compiled-scripts"),
        nwn_install_dir=None,  # TODO: Add your NWN install directory here!
        params=sys.argv[1:],
    )
