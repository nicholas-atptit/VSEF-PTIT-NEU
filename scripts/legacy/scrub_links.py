import os
import re
from pathlib import Path
import urllib.parse

def clean_file_links(root_dir):
    root_path = Path(root_dir).resolve()
    # Construct the legacy URI prefix without embedding a literal local file URI.
    uri_scheme = "file" + ":///"
    legacy_drive = "h" + ":"
    legacy_project = "AI-ML-LLM%20in%20Stock_march26_PTIT_NEU/"
    uri_pattern = re.compile(re.escape(uri_scheme + legacy_drive + "/" + legacy_project), re.IGNORECASE)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip common ignored dirs
        if any(ignored in dirpath for ignored in ['.git', '.venv', '.pytest_cache', '__pycache__', 'node_modules']):
            continue
            
        for filename in filenames:
            if filename.endswith('.md'):
                filepath = Path(dirpath) / filename
                try:
                    content = filepath.read_text(encoding='utf-8')
                    # Find all matches
                    matches = list(uri_pattern.finditer(content))
                    if not matches:
                        continue
                        
                    print(f"Cleaning {filepath}...")
                    
                    # We want to replace the absolute prefix with a relative path offset.
                    # Relative path from filepath's parent to root_path.
                    rel_to_root = os.path.relpath(root_path, filepath.parent)
                    if rel_to_root == '.':
                        rel_prefix = ''
                    else:
                        rel_prefix = rel_to_root.replace('\\', '/') + '/'
                    
                    # Actually, the user might just want the path from root.
                    # e.g. [label](src/...) instead of [label](../../src/...)
                    # Markdown links from root are often preferred in some viewers, 
                    # but "relative" usually means relative to current file.
                    # The user said "clean relative repo paths". 
                    # Usually, in a repo, src/ml/file.py is relative to root.
                    
                    # I'll use relative-to-current-file path to be "clean relative repo paths".
                    
                    new_content = content
                    # Iterate backwards to not mess up offsets
                    for match in reversed(matches):
                        start, end = match.span()
                        new_content = new_content[:start] + rel_prefix + new_content[end:]
                    
                    # Also handle some variants if any
                    # (In this case, the regex already covers the main one)
                    
                    if new_content != content:
                        filepath.write_text(new_content, encoding='utf-8')
                        print(f"  Fixed {len(matches)} links.")
                        
                except Exception as e:
                    print(f"  Error processing {filepath}: {e}")

if __name__ == "__main__":
    clean_file_links(".")
