import os

# Define file extensions to include as "code files"
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.html', '.htm', '.css',
    '.scss', '.json', '.xml', '.yml', '.yaml',
    '.java', '.c', '.cpp', '.h', '.cs', '.php',
    '.rb', '.go', '.rs', '.swift', '.kt', '.m',
    '.sh', '.bat', '.pl', '.sql', '.jsx', '.tsx',
    '.txt'  # Include .txt files specially
}

OUTPUT_FILE = "codebase_dump.txt"
SCRIPT_NAME = os.path.basename(__file__)
EXCLUDED_FILES = {SCRIPT_NAME, OUTPUT_FILE}
EXCLUDED_DIRS = {'etc', '__pycache__'}

def is_code_file(filename):
    return any(filename.endswith(ext) for ext in CODE_EXTENSIONS)

def should_exclude_dir(dirpath):
    parts = os.path.relpath(dirpath, os.getcwd()).split(os.sep)
    return any(part in EXCLUDED_DIRS for part in parts)

def write_folder_structure(root_path, output_file):
    output_file.write("This is a code base dumped into a single text file, divided into (1) folder and file structure and (2) the code itself.\n\n1) Folder and file structure of the code base:\n\n")
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        if should_exclude_dir(dirpath):
            continue

        depth = dirpath.replace(root_path, '').count(os.sep)
        indent = '    ' * depth
        output_file.write(f"{indent}{os.path.basename(dirpath)}/\n")
        for f in filenames:
            if f in EXCLUDED_FILES:
                continue
            output_file.write(f"{indent}    {f}\n")
    output_file.write("\n")
    output_file.write("2) Code per file:\n\n")

def write_code_files(root_path, output_file):
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        if should_exclude_dir(dirpath):
            continue

        for filename in filenames:
            if filename in EXCLUDED_FILES:
                continue
            if is_code_file(filename):
                file_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(file_path, root_path)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        if filename.endswith('.txt'):
                            lines = []
                            for _ in range(10):
                                line = f.readline()
                                if not line:
                                    break
                                lines.append(line)
                            content = ''.join(lines)
                            output_file.write(f"## File: {relative_path} (10 lines sample)\n\n")
                            output_file.write(content + "\n\n")
                        else:
                            content = f.read()
                            output_file.write(f"## File: {relative_path}\n\n")
                            output_file.write(content + "\n\n")
                except Exception as e:
                    output_file.write(f"## File: {relative_path} (could not read: {e})\n\n")

def main():
    root_path = os.getcwd()
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as output_file:
        write_folder_structure(root_path, output_file)
        write_code_files(root_path, output_file)

if __name__ == "__main__":
    main()

