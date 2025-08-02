import os


project = "TS_MTL"

# 2) Folders
folders = [
    "configs",
    f"src/{project}/models",
    f"src/{project}/trainers",
    f"src/{project}/data",
    f"src/{project}/utils",
    "scripts",
    "tests",
    "notebooks",
]

# 3) Files as placeholders
files = [
    "README.md",
    ".gitignore",
    "configs/default.yaml",
    f"src/{project}/__init__.py",
    f"src/{project}/cli.py",
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file in files:
    dirpath = os.path.dirname(file)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    open(file, "a").close()

print(f"Scaffolded '{project}' with model+trainer slots.")
