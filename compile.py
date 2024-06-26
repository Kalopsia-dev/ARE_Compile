from collections import defaultdict # For inverting the script index.
from hashlib import sha256          # For hashing script files.
from glob import glob               # For batch file operations.
import json                         # For storing script hashes.
import os                           # For OS-level operations.
import re                           # For script parsing.
import shutil                       # For file operations.
import sys                          # For command line arguments.
import time                         # For compile time measurement.
import subprocess                   # For calling the compiler.

class ScriptIndex:
    '''An index that stores references to unique Script objects and resolves their dependencies.'''

    # Default location for the script index file.
    PATH = os.path.join(os.path.split(__file__)[0], 'tmp', 'script_index.json')

    # Regular expressions for parsing script files.
    regex_comments = re.compile(r'/\*([\s\S]*?)?\*/',                       re.MULTILINE)
    regex_includes = re.compile(r'^#include\s+"([^"]+)"(?![^"\n]*//)',      re.MULTILINE)
    regex_main     = re.compile(r'^\s*void\s+main\s*\(\s*\)',               re.MULTILINE)
    regex_sc       = re.compile(r'^\s*int\s+StartingConditional\s*\(\s*\)', re.MULTILINE)

    def __init__(self, directory: str) -> None:
        '''Initialises a script index object and its children Scripts using a file name.'''
        print('Indexing scripts...')
        self.directory = directory
        self.scripts = dict()

        # Ensure the script index parent directory exists.
        os.makedirs(os.path.split(ScriptIndex.PATH)[0], exist_ok=True)

        # Parses all scripts in the script directory, genearting a script object for each.
        {Script(file, self) for file in os.listdir(self.directory)}

        # Flattens all includes in the scripts storage, mapping each script to its dependencies.
        self.scripts = ScriptIndex.flatten_includes(self.scripts)

        # Inverts the script index so that each include file points to its children.
        self.includes = ScriptIndex.invert_index(self.scripts)

    def get_modified_scripts(self):
        '''Returns a set of all scripts that have been modified since the last compile.'''
        script_hashes = ScriptIndex.read_hash_index()
        return {script
                for script in self.scripts.values()
                if script.exists and (script.name not in script_hashes or script_hashes[script.name] != script.hash)}

    @staticmethod
    def flatten_includes(index : dict, *, prior : dict = None, depth : int = 0) -> dict:
        '''Recursion that flattens the include file hierarchy up to the maximum include file depth.'''
        # Combine every entry in the script index with its corresponding include's includes.
        if depth < 10:
            [script.includes.update(include.includes)
             for script in index.values()
             for include in set(script.includes)]
            # Keep doing this until nothing changes.
            if index == prior: return index
            else: return ScriptIndex.flatten_includes(index, prior=index.copy(), depth=depth+1)
        else:
            print('Warning: Maximum include depth exceeded. Check for circular dependencies!')
            return index

    @staticmethod
    def invert_index(index : dict) -> dict:
        '''Returns an inverted script index where include file point at their children files.'''
        inverted = defaultdict(set)
        [inverted[include].add(script)
         for script in index.values()
         for include in script.includes]

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

    def generate_hash_index(self) -> dict:
        '''Returns a dictionary index of script hashes.
           The keys are script file names without extensions, the values are file hashes.'''
        hashes = {script.name: script.hash
                  for script in self.scripts.values()
                  if script.exists}
        return hashes

    @staticmethod
    def update_hash_index(modified: set) -> dict:
        hashes = ScriptIndex.read_hash_index()
        hashes.update({script.name: script.hash
                       for script in modified
                       if script.exists})
        return hashes

    @staticmethod
    def read_hash_index() -> dict:
        '''Loads the index of script hashes and returns them as a dictionary.
           The keys are script file names without extensions, the values are file hashes.'''
        try:
            with open(ScriptIndex.PATH, 'r') as hashes:
                return json.load(hashes)
        except: return dict()

    @staticmethod
    def write_hash_index(script_hashes : dict) -> None:
        '''Writes the given index of script hashes to a default location.'''
        try:
            with open(os.path.expanduser(ScriptIndex.PATH), 'w') as outfile:
                outfile.write(json.dumps(script_hashes))
        except Exception as e:
            print(f'Error: Unable to store script hashes ({e})')

    def delete_hash_index() -> dict:
        '''Deletes the index of script hashes.'''
        try: os.remove(ScriptIndex.PATH)
        except: return

class Script:
    '''A data class for script files that stores an individual script's name, path, hash, and include relations.'''

    # File suffixes for script files.
    NSS = '.nss' # Source script file.
    NCS = '.ncs' # Compiled script file.

    def __init__(self, name: str, script_index: ScriptIndex):
        '''Initialises a script object and its children Scripts using a file name.'''
        # Initialise the script name and path, then check if the file exists.
        self.name   = Script.normalise_script_name(name)
        self.path   = os.path.join(script_index.directory, f'{self.name}{Script.NSS}')
        self.exists = os.path.isfile(self.path)

        # Initialise class variables.
        self.is_include : bool = False
        self.has_main   : bool = False
        self.includes   : set  = set()
        self.hash       : int  = 0

        # Add this script to the script index if it's not included yet.
        if self.name not in script_index.scripts:
            script_index.scripts[self.name] = self

            # If this script points at an existing file, parse its includes recursively.
            if self.exists:
                with open(self.path, 'rb') as file:
                    # First generate a hash for later use.
                    contents  = file.read()
                    self.hash = int(sha256(contents).hexdigest(), 16)
                    # Afterwards, decode the file, strip block comments and analyse the remaining file contents.
                    contents       = re.sub(script_index.regex_comments, '', contents.decode(encoding = 'ISO-8859-1'))
                    self.has_main  = re.search(script_index.regex_main, contents) or re.search(script_index.regex_sc, contents)
                    self.includes  = {Script(include, script_index) if include not in script_index.scripts else script_index.scripts[include]
                                      for include in re.findall(script_index.regex_includes, contents)}

    def __hash__(self):
        '''Returns this script's hash value.'''
        return self.hash

    def __eq__(self, other):
        '''Compares scripts by their hash values.'''
        if not isinstance(other, Script): return False
        return self.hash == other.hash and self.path == other.path

    def __repr__(self):
        '''Returns a string representation of the script.'''
        return f'{self.name} <{"Script" if not self.is_include else "Include"} with {len(self.includes)} include(s)>'

    @staticmethod
    def normalise_script_name(name: str):
        return os.path.splitext(name)[0].lower()

class Compiler:
    '''A compiler that intelligently compiles scripts based on their include relations and hash changes.'''

    def __init__(self, params: list, compiler : str, script_dir: str, output_dir: str, nwn_install_dir : str = '', nwn_user_dir : str = '', *, secondary_output_dir = None):
        '''Initialises a compiler object using a script index.'''
        # Check whether the provided paths exist. If output directories are missing, create them.
        if not os.path.exists(script_dir):                          print(f'Error: Unable to locate script directory at "{script_dir}"')      ; exit(1)
        if nwn_user_dir and not os.path.exists(nwn_user_dir):       print(f'Error: Unable to locate NWN home directory at "{nwn_user_dir}"')  ; exit(1)
        if nwn_install_dir and not os.path.exists(nwn_install_dir): print(f'Error: Unable to locate NWN installation at "{nwn_install_dir}"') ; exit(1)
        if secondary_output_dir: os.makedirs(secondary_output_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Store the current time to calculate the total execution time later.
        self.start_time = time.time()

        # Store the given parameters and directories.
        self.nwn_install_dir = nwn_install_dir
        self.nwn_user_dir = nwn_user_dir
        self.output_dir = output_dir
        self.compiler = compiler

        # Compile all scripts if the output directory is empty or there is no hash index.
        output_files  = len(glob(os.path.join(self.output_dir, f'*{Script.NCS}')))
        script_hashes = ScriptIndex.read_hash_index()
        if params != ['all'] and not (output_files and script_hashes):
            print('All scripts will be compiled to initialise the index.', end='\n\n')
            params = ['all']

        # Generate a script index to analyse the include structure.
        self.script_index = ScriptIndex(script_dir)

        # Determine the compile mode based on the given parameters and availability of a hash index.
        if params:
            # If we have an argument, use it to determine what to compile.
            param = Script.normalise_script_name(params[0])
            if param == 'all':     self.compile_all()
            elif param[-1] == '*': self.compile_wildcard(param)
            else:                  self.compile_script(param)
        else: self.compile_modified()

        if secondary_output_dir:
            # If we're live compiling, copy scripts compiled since start_time to the server's development folder.
            for script in os.listdir(self.output_dir):
                if os.path.getmtime(os.path.join(self.output_dir, script)) > self.start_time:
                    shutil.copyfile(os.path.join(self.output_dir, script), os.path.join(secondary_output_dir, script))

    def run_script_comp(self, path: str, *, silent = False) -> bool:
        '''Runs nwn_script_comp on a given script file or wildcard path. Returns True if the operation was successful.'''
        compile_command = [self.compiler,                                  # Prepare nwn_script_comp calls. The flags are as follows:
                           '-c',                                           # : Compile multiple files and/or directories.
                           '--quiet',                                      # : Turn off all logging, except errors and above
                           '--max-include-depth=64',                       # : Maximum include depth [default: 16]
                           '--dirs', self.script_index.directory]          # : Load comma-separated directories [default: ]
        if self.nwn_install_dir: # Allow overriding the default NWN directories.
            compile_command.extend(['--root', self.nwn_install_dir])       # : Override NWN root (autodetection is attempted)
        if self.nwn_user_dir:
            compile_command.extend(['--userdirectory', self.nwn_user_dir]) # : Override NWN user directory (autodetection is attempted)
        remaining = None # Stores remaining scripts to be compiled if the command line was too long.

        # Check if the path is a list of paths, a directory, or a file.
        if isinstance(path, list):
            # If it is a list of paths, we're compiling multiple files within the same directory.
            if not silent: print('Compiling...', end='\n\n')
            # Windows has a command line limit of 32768 characters. If we exceed this, we'll need to compile in batches.
            WIN_CMD_LIMIT = 32768 - 768 if sys.platform == 'win32' else 0
            if not WIN_CMD_LIMIT or len(' '.join(compile_command + path)) < WIN_CMD_LIMIT:
                # This is a UNIX system or the command line is short enough. We can compile all files at once.
                compile_command.extend(path)
            else:
                # The command line is too long. We'll have to compile in batches.
                while len(" ".join(compile_command)) < WIN_CMD_LIMIT:
                    # Add files to the command line until it's dangerously close to the limit.
                    compile_command.append(path.pop(0))
                # Store the remaining script list for later.
                remaining = path.copy()
            path = os.path.dirname(path[0])
            isdir = True
        else:
            # Otherwise, we're compiling a single file or directory.
            isdir = os.path.isdir(path)
            if isdir:
                # We're compiling all scripts within the given directory.
                compile_command.append(path)
                if not silent: print('Compiling...', end='\n\n')
            else:
                # We're compiling a single script file. Only append the given file name.
                compile_command.remove('-c')
                compile_command.append(f'{path}')
                file_name = os.path.splitext(os.path.basename(path))[0]
                if not silent: print(f'Compiling: {file_name}{Script.NSS}', end='\n\n')

        # Run nwn_script_comp on the given path and analyse the output.
        with subprocess.Popen(compile_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as compile_process:
            # Check the error channel of the process output.
            output = compile_process.communicate()[1]
            # Try to decode it using different encodings. This improves compatibility with different systems.
            encodings = ['utf-8', 'windows-1250', 'windows-1252', 'ISO-8859-1']
            for encoding in encodings:
                try:
                    # Decode the output, then isolate the error message.
                    error_msg = output.decode(encoding=encoding)
                    error_msg = error_msg.split(sep=path, maxsplit=1)[-1].strip(' :').strip()
                    break
                except UnicodeDecodeError: continue
            if error_msg:
                # If the operation was unsuccessful, return False.
                print(f'{error_msg}\n\nStopping processing on first error.\n\n1 error; see above for context.\n\nProcessing aborted.')
                return False

        # If we have remaining files and did not fail in this batch, continue compiling instead of proceeding.
        if remaining: return self.run_script_comp(remaining, silent=True)

        # If we got here, the operation was successful.
        if isdir:
            # Move the generated .ncs files to our output folder.
            print('Moving script(s) to output folder...')
            all(shutil.move(file, os.path.join(self.output_dir, os.path.basename(file)))
                for file in glob(os.path.join(path, f'*{Script.NCS}')))
            print('Success!', end='\n\n')
        else:
            # Move the generated .ncs file to our output folder. We'll need to replace the extension.
            shutil.move(f'{path[:-4]}{Script.NCS}', os.path.join(self.output_dir, f'{file_name}{Script.NCS}'))
        print(f'Total Execution time = {time.time() - self.start_time:.4f} seconds', end='\n\n')
        return True

    def find_related_scripts(self, scripts: set) -> set:
        '''Returns a set of all scripts (except includes) affected by changes to the given set of scripts.'''
        # Handle single script inputs.
        if isinstance(scripts, Script): scripts = {scripts}
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

    def compile_script(self, script_name : str) -> None:
        '''Compiles an individual script file.'''
        script = self.script_index.scripts[script_name] if script_name in self.script_index.scripts else None
        if script and script.exists:
            # Check if this is an include file.
            if not script.is_include:
                # It is not. Generate the given script's hash. We'll remember it if this compile is successful.
                script_hashes = ScriptIndex.update_hash_index({script})
                # Compile the file. If the operation is successful, update the hash index.
                if self.run_script_comp(script.path): ScriptIndex.write_hash_index(script_hashes)
            else:
                # It is an include file. Compile all dependencies.
                print('Include file detected. Checking dependencies...', end='\n\n')
                to_compile = self.find_related_scripts(script)
                if to_compile:
                    print(f'{len(to_compile)} script(s) include {script.name}.', end='\n\n')
                    self.compile_set(to_compile, script)
                else: print(f'No scripts include {script.name}.')
        else: print(f'Error: Unable to find {script_name}{Script.NSS}')

    def compile_all(self) -> None:
        '''Compiles all scripts in the main script directory and copies the results to the output directory.'''
        # Clear all files within the output directory. No need to keep them if we're going to compile all scripts.
        Compiler.__clean_directory__(self.script_index.directory)
        Compiler.__clean_directory__(self.output_dir)
        ScriptIndex.delete_hash_index()
        # Load the hashes of all scripts in the script directory. We'll remember them if this compile is successful.
        script_hashes = self.script_index.generate_hash_index()
        # Now batch compile all script files with run_script_comp. If successful, update the hash index.
        if self.run_script_comp(self.script_index.directory): ScriptIndex.write_hash_index(script_hashes)
        # Clear all remaining files.
        Compiler.__clean_directory__(self.script_index.directory)

    def compile_set(self, scripts : set, modified : set) -> None:
        '''Compiles all scripts in a given set within the temporary folder and copies the results to the output directory, if successful.'''
        # Handle single script inputs.
        if isinstance(scripts,  Script): scripts  = {scripts}
        if isinstance(modified, Script): modified = {modified}
        # Load the script hashes of all scripts in the given set. We'll remember them if this compile is successful.
        script_hashes = ScriptIndex.update_hash_index(modified)
        Compiler.__clean_directory__(self.script_index.directory)
        # Check if the operation was successful.
        if self.run_script_comp([script.path for script in scripts]): ScriptIndex.write_hash_index(script_hashes)
        # Clear all remaining files.
        Compiler.__clean_directory__(self.script_index.directory)

    def compile_modified(self) -> None:
        '''Compile all modified scripts in the script directory.'''
        # Generate a set of new and modified scripts and check what needs to be compiled.
        modified = self.script_index.get_modified_scripts()
        to_compile = self.find_related_scripts(modified)
        if modified and to_compile:
            print(f'{len(modified)} change(s) found. {len(to_compile)} affected script(s) will be compiled.', end='\n\n')
            self.compile_set(to_compile, modified)
        else: print('All scripts are up to date.')

    def compile_wildcard(self, script_name : str) -> None:
        '''Compiles all scripts matching the given wildcard.'''
        # Compile all scripts if a blank wildcard is given.
        if script_name == '*': return self.compile_all()
        # Check if the wildcard matches any scripts.
        regex = re.compile(script_name.replace('*', '.*'), re.IGNORECASE)
        matches = {script for script in self.script_index.scripts.values()
                   if script.exists and regex.match(script.name)}
        to_compile = self.find_related_scripts(matches)
        if matches and to_compile:
            # Print a list of all matches, then compile the affected scripts.
            if len(matches) < 30:
                print(f'\n{len(matches)} match(es) found:')
                [print(f'- {script_name}{Script.NSS}')
                 for script_name in sorted([script.name for script in matches])]
            else: print(f'\n{len(matches)} match(es) found.')
            print(f'\n{len(to_compile)} related script(s) will be compiled.', end='\n\n')
            self.compile_set(to_compile, matches)
        else: print('No matches found.')

    @staticmethod
    def __clean_directory__(directory: str) -> None:
        '''Removes all compiled script files from a given directory.'''
        all(not os.remove(file) for file in glob(os.path.join(directory, f'*{Script.NCS}')))

if __name__ == '__main__':
    # Create a script index and compile based on the given command-line parameters. We'll look for nwn_script_comp in the same directory.
    compiler_path = os.path.join(os.path.split(__file__)[0], 'nwn_script_comp.exe' if sys.platform == 'win32' else 'nwn_script_comp')
    if not os.path.exists(compiler_path):
        # If the compiler is not found, we cannot proceed. Exit with an error.
        print(f'Error: Unable to locate nwn_script_comp at "{compiler_path}"')
        exit(1)

    Compiler(
        script_dir=os.path.join(os.getcwd(), 'scripts'),
        output_dir=os.path.join(os.getcwd(), 'compiled-scripts'),
        compiler=compiler_path,
        params=sys.argv[1:],
    )
