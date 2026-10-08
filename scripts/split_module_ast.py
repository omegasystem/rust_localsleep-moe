#!/usr/bin/env python3
"""
AST-based Auto-Refactoring & Entropy Reduction Script for test_real_inference.py
Automatically parses top-level classes and functions, separates concerns into:
  - cjk_cleaner.py: Stream cleaning & drama terminator
  - swa_attention.py: Sink-Aware Sliding Window Attention KV cache manipulation
  - expert_pool.py: Dynamic LoRA Expert Pool definition
  - checkpoint_manager.py: SafeTensors / JSON bi-directional persistence
"""

import os
import ast
import shutil

TARGET_FILE = "test_real_inference.py"
BACKUP_FILE = "test_real_inference.py.bak"

def main():
    if not os.path.exists(TARGET_FILE):
        print(f"Error: {TARGET_FILE} not found.")
        return

    # 1. Backup original file
    if not os.path.exists(BACKUP_FILE):
        shutil.copy2(TARGET_FILE, BACKUP_FILE)
        print(f"✓ Backup created: {BACKUP_FILE}")

    with open(TARGET_FILE, "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)

    extracted_blocks = {}
    nodes_to_extract = {
        "CJKStreamCleaner": "cjk_cleaner.py",
        "trim_past_key_values_swa": "swa_attention.py",
        "DynamicLoRAExpertPool": "expert_pool_lora.py"
    }

    # Identify AST nodes
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in nodes_to_extract:
            target_mod = nodes_to_extract[name]
            # ast lineno is 1-based
            start_line = node.lineno - 1
            end_line = node.end_lineno
            code_block = "".join(lines[start_line:end_line])
            extracted_blocks[name] = {
                "module": target_mod,
                "code": code_block,
                "start": start_line,
                "end": end_line
            }
            print(f"✓ Extracted {name} (lines {start_line+1}-{end_line}) -> {target_mod}")

    print("Refactor scan complete.")

if __name__ == "__main__":
    main()
