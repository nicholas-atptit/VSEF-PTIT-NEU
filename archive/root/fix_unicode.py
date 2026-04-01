import os
from pathlib import Path

def fix_unicode(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py") or file.endswith(".md"):
                file_path = Path(root) / file
                try:
                    content = file_path.read_text(encoding="utf-8")
                    if "→" in content:
                        print(f"Fixing {file_path}")
                        new_content = content.replace("→", "->")
                        file_path.write_text(new_content, encoding="utf-8")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    fix_unicode("src/ml")
    fix_unicode("scripts")
