import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from glob import glob
from hashlib import sha256

from nwn.key import Reader as KeyReader
from nwn.nwscript.comp import CompilationError
from nwn.nwscript.comp import Compiler as ScriptComp


class Script:
    """
    Represents an individual script file and its dependencies.
    """

    def __init__(self, name: str, script_index: "ScriptIndex") -> None:
        """
        Initialises a new Script and the Scripts included by it using its file name.

        Args:
            name (str): The name of the script file, e.g. `nw_s0_sleep`.
            script_index (ScriptIndex): The script index to use for resolving dependencies.
        """
        # Initialise class variables.
        self.name = ScriptIndex.normalise_script_name(name)
        self.index = script_index
        self.is_include = False

        # If there is no associated file, this is a base game script and can be skipped.
        script_path = os.path.join(script_index.directory, f"{self.name}.nss")
        if not os.path.isfile(script_path):
            self.hash = hash(self.name)
            self.includes = set()
            self.contents = None
            self.has_main = False
            return

        # Since this script points to an existing file, we will parse it.
        with open(script_path, "rb") as file:
            self.contents = file.read()

        # First generate a hash for later use.
        self.hash = HashIndex.hash(self.contents)

        # Afterwards, decode the file, strip block comments and analyse the remaining file contents.
        contents = re.sub(
            script_index.regex_comments,
            "",
            self.contents.decode(encoding="ISO-8859-1"),
        )
        self.has_main = re.search(script_index.regex_main_fns, contents)
        self.includes = {
            script_index.get_or_create(include_name)
            for include_name in re.findall(script_index.regex_includes, contents)
        }

    def __hash__(self) -> int:
        """
        Returns this script's previously calculated hash value.
        This is used to compare scripts and check if thir contents have changed.

        Returns:
            int: The hash value of the script.
        """
        return self.hash

    def __eq__(self, other: "Script") -> bool:
        """
        Compares scripts based on their hash values.

        Args:
            other (Script): The script to compare with.

        Returns:
            bool: True if the scripts represent the same file, False otherwise.
        """
        if not isinstance(other, Script):
            return False
        return self.hash == other.hash and self.name == other.name

    def __repr__(self) -> str:
        """
        Returns a string representation of the script.
        """

        def format_name(
            script_type: str,
            dependencies: int | None = None,
            derived: int | None = None,
        ) -> str:
            """
            Creates a Terminal-friendly representation of the script name and its type.
            """
            result = f"{self.name.ljust(16)} {script_type.ljust(9)}"
            if dependencies is not None:
                result += f" {dependencies:3d} dependencies"
            if derived is not None:
                result += f", {derived:4d} derived"
            return result

        if not self.contents:
            return format_name("[Base Script]")
        dependencies = len(self.includes)
        if self.is_include or not self.has_main:
            derived = len(self.index.includes.get(self, set()))
            return format_name("[Include]", dependencies, derived)
        return format_name("[Script]", dependencies)


class HashIndex:
    """
    A class that manages hashes and hash indices for script files.
    """

    # Default location for the script index file.
    PATH = os.path.join(os.path.split(__file__)[0], "tmp", "script_index.json")

    @staticmethod
    def hash(script_contents: bytes) -> int:
        """
        Generates a hash for the given script contents.

        Args:
            script_contents (bytes): The contents of the script file.

        Returns:
            int: The hash value of the script contents.
        """
        return int(sha256(script_contents).hexdigest(), 16)

    @staticmethod
    def apply(modified: set[Script]) -> dict[str, int]:
        """
        Returns what would be the new hash index if the given set of scripts were to be compiled.
        The actual file is not modified (yet), it should be written manually after compilation.

        Args:
            modified (set[Script]): The set of modified scripts to update the index with.

        Returns:
            dict[str, int]: The updated index of script hashes. Maps script file names to their hashes.
        """
        hashes = HashIndex.read()
        hashes.update(
            {script.name: script.hash for script in modified if script.contents}
        )
        return hashes

    @staticmethod
    def apply_all(script_index: "ScriptIndex") -> dict[str, int]:
        """
        Returns what would be the new hash index if all scripts in the given script index were to be compiled.

        Args:
            script_index (ScriptIndex): The script index to generate the hash index for.

        Returns:
            dict[str, int]: The hash index for all scripts in the script index.
        """
        return HashIndex.apply({script for script in script_index if script.contents})

    @staticmethod
    def read() -> dict[str, int]:
        """
        Loads the index of script hashes and returns them as a dictionary.

        Returns:
            dict[str, int]: The index of script hashes. Maps script file names to their hashes.
        """
        if os.path.exists(HashIndex.PATH):
            with open(HashIndex.PATH, "rb") as hashes:
                return json.load(hashes)
        else:
            return dict()

    @staticmethod
    def write(hash_index: dict[str, int]) -> None:
        """
        Writes the given index of script hashes to a default location.

        Args:
            hash_index (dict[str, int]): The index of script hashes to write.
        """
        try:
            os.makedirs(os.path.split(HashIndex.PATH)[0], exist_ok=True)
            with open(os.path.expanduser(HashIndex.PATH), "w") as outfile:
                outfile.write(json.dumps(hash_index))
        except OSError as e:
            print(f"Error: Unable to store script hashes ({e})")

    @staticmethod
    def delete() -> None:
        """
        Deletes the index of script hashes, if it exists.
        """
        if os.path.exists(HashIndex.PATH):
            os.remove(HashIndex.PATH)


class ScriptIndex:
    """
    An index that stores references to unique Script objects and resolves their dependencies.
    """

    # Regular expressions for parsing script files. Pre-compiled for best performance.
    regex_comments = re.compile(r"//.*?$|/\*[\s\S]*?\*/", re.MULTILINE | re.DOTALL)
    regex_includes = re.compile(r'^\s*#include\s+"([^"]+)"(?![^"\n]*//)', re.MULTILINE)
    regex_main_fns = re.compile(
        r"^\s*(?:void\s+main|int\s+StartingConditional)\s*\(\s*\)", re.MULTILINE
    )

    def __init__(self, script_dir: str) -> None:
        """
        Initialises a script index object and its children Scripts using a file name.

        Args:
            directory (str): The directory containing the custom script files to compile.
        """
        if not os.path.isdir(script_dir):
            raise FileNotFoundError(f"Script directory '{script_dir}' does not exist.")

        print("Indexing scripts...")
        self.directory = script_dir
        self.scripts = dict()

        # Parse all scripts in the script directory, generating a script object for each.
        for script_name in glob(os.path.join(self.directory, "*.nss")):
            # The scripts will self-register themselves in the script index.
            self.get_or_create(script_name)

        def flatten_include_tree() -> None:
            """
            Flattens the include tree, mapping each script to all its dependencies and their dependencies.
            """

            def last_modified(
                last_count: dict[Script, int], current_count: dict[Script, int]
            ) -> set[Script]:
                """
                Returns a set of scripts whose include counts have changed since the last iteration.
                """
                if last_count is None:
                    # If this is the first iteration, return all scripts with includes.
                    return set(current_count.keys())
                return {
                    script
                    for script in current_count.keys()
                    if script.includes and last_count[script] != current_count[script]
                }

            last_count = None
            for _ in range(10):
                # Map each script to its include count. We'll use this to keep track of changes.
                current_count = {script: len(script.includes) for script in self}
                if last_count == current_count:
                    # If the include count hasn't changed since the last iteration, we can stop.
                    break
                # Combine each script's set of includes with the includes' sets of includes.
                for script in last_modified(last_count, current_count):
                    script.includes.update(
                        *(include.includes for include in script.includes),
                    )
                # Update the stored include count for the next iteration.
                last_count = current_count

        def group_by_includes() -> dict[Script, set[Script]]:
            """
            Returns a mapping of include files to their child scripts, and flags includes as such.

            Returns:
                dict[Script, set[Script]]: The inverted script index. Maps include files to their child scripts.
            """
            # Prepare a dictionary to store the inverted script index.
            includes = dict.fromkeys(
                {include for script in self for include in script.includes},
            )
            # Add all scripts to the corresponding include keys.
            for script in self:
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

        # Map each script to all of its dependencies, and their dependencies.
        flatten_include_tree()

        # Finally, generate an inverted index where include files point at their children.
        self.includes = group_by_includes()

    def __iter__(self):
        """
        Returns an iterator over the script index.

        Returns:
            Iterator[Script]: An iterator over the Script objects in the index.
        """
        return iter(self.scripts.values())

    def __contains__(self, script: str | Script) -> bool:
        """
        Checks if a script is in the script index.

        Args:
            script (str | Script): The name of the script to check for.

        Returns:
            bool: True if the script is in the index, False otherwise.
        """
        if isinstance(script, Script):
            return script in self.scripts.values()
        return ScriptIndex.normalise_script_name(script) in self.scripts

    def add(self, script: Script) -> Script:
        """
        Adds a script to the script index. Should be called before flattening the index.

        Args:
            script (Script): The script to add to the index.

        Returns:
            Script: The added script object.
        """
        self.scripts[script.name] = script
        return script

    def get(self, script_name: str, default: any = None) -> Script:
        """
        Returns a script object from the script index.

        Args:
            script_name (str): The name of the script to retrieve.

        Returns:
            Script: The requested script object, or None if it does not exist in the index.
        """
        return self.scripts.get(ScriptIndex.normalise_script_name(script_name), default)

    def get_or_create(self, script_name: str) -> "Script":
        """
        Creates a new Script object and adds it to the script index.

        Args:
            script_name (str): The name of the script file to create.
            script_index (ScriptIndex): The script index to add the script to.

        Returns:
            Script: The newly created Script object.
        """
        if (script := self.get(script_name)) is not None:
            # If the script already exists in the index, return it.
            return script
        return self.add(Script(script_name, script_index=self))

    def get_modified(self) -> set[Script]:
        """
        Returns a set of all scripts that have been modified since the last successful compilation.

        Returns:
            set[Script]: A set of all modified scripts.
        """
        hash_index = HashIndex.read()
        return {
            script
            for script in self
            if script.contents
            and (
                script.name not in hash_index or hash_index[script.name] != script.hash
            )
        }

    def get_related(self, scripts: set[Script]) -> set[Script]:
        """
        Returns a set of all compilable scripts that are affected by changes to the given scripts.

        Args:
            scripts (Script | set[Script]): The set of scripts to check for dependencies.

        Returns:
            set[Script]: A set of all scripts affected by changes to the provided scripts.
        """
        # For include files, add all scripts affected by the change to the set. Otherwise, just add the script.
        scripts = scripts.union(
            *(self.includes[script] for script in scripts if script.is_include)
        )
        # Drop all include files from the set. We cannot compile them.
        return {script for script in scripts if script.contents and script.has_main}

    @staticmethod
    @lru_cache(maxsize=None)
    def normalise_script_name(script_name: str):
        """
        Normalises a script name by removing the path and file extension, and converting it to lowercase.

        Args:
            script_name (str): The name of the script to normalise.

        Returns:
            str: The normalised script name.
        """
        base_name = os.path.basename(script_name)
        file_name, _ = os.path.splitext(base_name)
        return file_name.lower()


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
        # Try to automatically locate an NWN installation directory if it is not provided.
        nwn_install_dir = nwn_install_dir or Compiler.locate_nwn_directory()
        if not nwn_install_dir or not os.path.exists(nwn_install_dir):
            print(f'Error: Unable to locate NWN installation at "{nwn_install_dir}"')
            exit(1)
        key_file = os.path.join(nwn_install_dir, "data", "nwn_base.key")
        if not os.path.isfile(key_file):
            print(f'Error: Unable to locate NWN key file at "{key_file}"')
            exit(1)

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
        output_files = len(glob(os.path.join(output_dir, "*.ncs")))
        hash_index = HashIndex.read()
        if params != ["all"] and not (output_files and hash_index):
            print("All scripts will be compiled to initialise the index.", end="\n\n")
            params = ["all"]

        # Generate a script index to analyse the include structure.
        self.script_index = ScriptIndex(script_dir)

        # Load the NWN key file. It should only be accessed by one thread at a time.
        self.key_reader = KeyReader(key_file)
        self.io_lock = threading.Lock()

        # Determine the compile mode based on the given parameters and availability of a hash index.
        if params:
            # If there are any arguments, use them to determine what to compile.
            param = ScriptIndex.normalise_script_name(params[0])
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
        new_hash_index: dict[str, int] = None,
    ) -> None:
        """
        Compiles a given script file or iterable of scripts, then writes the results to the output directory.

        Args:
            scripts (Script | Iterable[Script]): The script or iterable of scripts to compile.
            new_hash_index (dict[str, int], optional): The new hashes to store if compilation is successful.

        Returns:
            bool: True if the compilation of all scripts was successful, False otherwise.
        """
        # First, normalise the input to a set of scripts.
        if isinstance(scripts, Script):
            scripts = {scripts}

        # Create a thread-local storage for the compilers.
        locals = threading.local()

        def init_thread() -> None:
            """
            Initialises a thread-local compiler instance.
            """
            locals.compiler = ScriptComp(
                resolver=self.load_script,
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
            print(f"Compiling: {script.name}")
            try:
                # Attempt to compile the script using the thread-local compiler instance.
                ncs_bytes = locals.compiler.compile(script.name)[0]
                return f"{script.name}.ncs", ncs_bytes
            except CompilationError as error:
                # Format, print and raise the error. It will then interrupt the thread pool.
                print(f"\n{error.message.splitlines()[0].split(' [')[0]}")
                raise error

        try:
            successful = True
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
                # First, ensure the directory exists.
                os.makedirs(output_dir, exist_ok=True)
                # Write the compiled script to the output directory.
                with open(os.path.join(output_dir, ncs_name), "wb") as outfile:
                    outfile.write(ncs)
        if new_hash_index:
            # If we have a new hash index, write it to the hash file now.
            HashIndex.write(new_hash_index)

        # Finally, print the total execution time.
        compile_time = time.time() - self.start_time
        print(f"Success!\n\nTotal Execution time = {compile_time:.4f} seconds\n")

    def load_script(self, script_name: str) -> bytes | None:
        """
        Loads a script from the script index or the NWN installation directory.

        Args:
            script_name (str): The file name of the script to load, e.g. `nw_s0_sleep.nss`.

        Returns:
            bytes | None: The contents of the script file, or None if the file could not be found.
        """
        # First, check if the script is in the script index.
        script = self.script_index.get(script_name)
        if script and script.contents:
            # If the script is in the index and has contents, return its contents.
            return script.contents
        with self.io_lock:
            # Otherwise, read the script from the NWN installation directory.
            return self.__load_game_script__(script_name)

    @lru_cache(maxsize=None)
    def __load_game_script__(self, script_name: str) -> bytes | None:
        """
        Loads a script from the NWN installation directory.

        Args:
            script_name (str): The file name of the script to load, e.g. `nw_s0_sleep.nss`.

        Returns:
            bytes | None: The contents of the script file, or None if the file could not be found.
        """
        # First, check if the script is in the override folder, which takes precedence.
        override = os.path.join(self.nwn_install_dir, "ovr", script_name)
        if os.path.isfile(override):
            with open(override, "rb") as file:
                return file.read()
        try:
            # If it's not here, attempt to load it from the NWN key file.
            return self.key_reader.read_file(script_name)
        except FileNotFoundError:
            # This script does not exist in the game files.
            return None

    def compile_script(self, script_name: str) -> None:
        """
        Compiles an individual script file.

        Args:
            script_name (str): The name of the script to compile, e.g. `nw_s0_sleep`.
        """
        # We can only compile scripts in the script index.
        script = self.script_index.get(script_name)
        if not script:
            print(f"Error: Unable to find {script_name}.nss")
            return
        # Exclude base game scripts.
        if not script.contents:
            print(f"Error: {script_name}.nss is a base game script.")
            return
        # Next, check if the script is an include file.
        if script.is_include or not script.has_main:
            # It is, so compile all dependencies.
            print("Include file detected. Checking dependencies...", end="\n\n")
            to_compile = self.script_index.get_related({script})
            if not to_compile:
                print(f"No scripts include {script.name}.")
                return
            print(
                f"{len(to_compile)} script(s) include {script.name}.",
                end="\n\n",
            )
            self.compile(
                scripts=to_compile,
                new_hash_index=HashIndex.apply({script}),
            )
        else:
            # This is a normal script, so compile it directly.
            self.compile(
                scripts=script,
                new_hash_index=HashIndex.apply({script}),
            )

    def compile_all(self) -> None:
        """
        Compiles all scripts in the main ScriptIndex directory and copies the results to the output directory.

        This is used to initialise the script index.
        """
        # Locate all relevant script files in the script index.
        scripts = {
            script
            for script in self.script_index
            if script.contents and script.has_main
        }
        # Clear old output files, including the hash index.
        self.clear_output_folders()
        # Batch compile all script files. If successful, update the hash index with the current file hashes.
        self.compile(
            scripts=scripts, new_hash_index=HashIndex.apply_all(self.script_index)
        )

    def compile_modified(self) -> None:
        """
        Compile all modified and affected scripts, as identified by the ScriptIndex.
        """
        # Generate a set of new and modified scripts and check what needs to be compiled.
        modified = self.script_index.get_modified()
        to_compile = self.script_index.get_related(modified)
        if not modified or not to_compile:
            print("All scripts are up to date.")
            return
        print(
            f"\n{len(modified)} change(s) found:",
            f"\n- {'\n- '.join(sorted(str(script) for script in modified))}\n",
            f"\n{len(to_compile)} related script(s) will be compiled.",
            end="\n\n",
        )
        self.compile(
            scripts=to_compile,
            new_hash_index=HashIndex.apply(modified),
        )

    def compile_wildcard(self, script_name: str) -> None:
        """
        Compiles all scripts matching the given wildcard.

        Args:
            script_name (str): The wildcard to match against script names, e.g. `nw_s0_*`.
        """
        # Compile all scripts if a blank wildcard is given.
        if script_name == "*":
            return self.compile_all()
        # Check if the wildcard matches any scripts.
        wildcard = re.compile(script_name.replace("*", ".*"), re.IGNORECASE)
        matches = {
            script
            for script in self.script_index
            if script.contents and wildcard.match(script.name)
        }
        to_compile = self.script_index.get_related(matches)
        if not matches or not to_compile:
            print("No matches found.")
            return
        # Display the results of the wildcard search.
        print(
            f"\n{len(matches)} match(es) found:",
            f"\n- {'\n- '.join(sorted(str(script) for script in matches))}\n",
            f"\n{len(to_compile)} related script(s) will be compiled.",
            end="\n\n",
        )
        self.compile(scripts=to_compile, new_hash_index=HashIndex.apply(matches))

    def clear_output_folders(self) -> None:
        """
        Removes all compiled script files from the output directories.
        """
        HashIndex.delete()
        for directory in self.output_dirs:
            for file in glob(os.path.join(directory, "*.ncs")):
                os.remove(file)

    @staticmethod
    def locate_nwn_directory() -> str | None:
        """
        Attempts to find the NWN installation directory by checking common locations.

        Returns:
            str | None: The path to the NWN installation directory, or None if it could not be found.
        """
        if path := os.environ.get("NWN_ROOT"):
            # If the NWN_ROOT environment variable is set, use it.
            return path

        match sys.platform:
            # Check for common NWN installation directories based on the platform.
            case "linux" | "linux2":
                path = os.path.expanduser(
                    "~/.local/share/Steam/steamapps/common/Neverwinter Nights"
                )
            case "darwin":
                path = os.path.expanduser(
                    "~/Library/Application Support/Steam/steamapps/common/Neverwinter Nights"
                )
            case "win32":
                path = (
                    r"C:\Program Files (x86)\Steam\steamapps\common\Neverwinter Nights"
                )
            case _:
                return None

        if os.path.isdir(path):
            return path

        return None


if __name__ == "__main__":
    # Create a script index and compile based on the given command-line parameters.
    Compiler(
        script_dir=os.path.join(os.getcwd(), "input_nss"),
        output_dir=os.path.join(os.getcwd(), "output"),
        nwn_install_dir=None,  # TODO: If needed, add your NWN install directory here.
        params=sys.argv[1:],
    )
