import argparse
import subprocess
import pathlib
import json
import shutil
import os
from datetime import datetime
from typing import List, Dict, Optional

class PandoLaneManager:
    """
    Pando-Lite: Local Parallel Git Worktree & L0->L1->L2 Layered Merge Manager.
    Architecture:
    L0 (Experimental) -> L1 (Consolidated) -> L2 (Public Projection)
    """
    def __init__(self, root_dir: str = "."):
        self.root = pathlib.Path(root_dir).resolve()
        self.worktrees_dir = self.root / ".worktrees"
        self.worktrees_dir.mkdir(exist_ok=True)
        self.config_file = self.worktrees_dir / "pando_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.registry, f, indent=4)

    def _git(self, args: List[str], cwd: pathlib.Path = None) -> str:
        cmd = ["git"] + args
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    def _get_commit_diff(self, base: str, target: str) -> str:
        # Returns "Ahead X, Behind Y"
        try:
            rev_list = self._git(["rev-list", "--left-right", "--count", f"{base}...{target}"])
            behind, ahead = rev_list.split()
            return f"Ahead {ahead}, Behind {behind}"
        except Exception:
            return "Unknown"

    def status(self):
        print(f"{'Lane Name':<20} | {'Branch':<20} | {'Status':<10} | {'Diff to Trunk'}")
        print("-" * 70)
        
        trunk_branch = "private_main" # Default trunk
        
        for lane, info in self.registry.items():
            lane_path = self.worktrees_dir / lane
            if not lane_path.exists():
                continue
            
            # Check if clean
            is_clean = "Clean" if not self._git(["status", "--porcelain"], cwd=lane_path) else "Dirty"
            diff = self._get_commit_diff(trunk_branch, info['branch'])
            
            print(f"{lane:<20} | {info['branch']:<20} | {is_clean:<10} | {diff}")

    def spawn(self, lane_name: str, base: str, branch: Optional[str] = None):
        target_path = self.worktrees_dir / lane_name
        if target_path.exists():
            print(f"Error: Lane {lane_name} already exists.")
            return

        target_branch = branch or f"agent/{lane_name}"
        
        print(f"Spawning lane {lane_name} from {base} -> {target_branch}...")
        try:
            # Git worktree add -b <branch> <path> <base>
            self._git(["worktree", "add", "-b", target_branch, str(target_path), base])
            
            # Zero-copy acceleration: Symlink heavy caches
            # We assume the main root has the .venv_rust or target folders
            for cache_dir in [".venv_rust", "target", ".cargo"]:
                src_cache = self.root / cache_dir
                dst_cache = target_path / cache_dir
                if src_cache.exists() and src_cache.is_dir():
                    if dst_cache.exists():
                        shutil.rmtree(dst_cache)
                    os.symlink(src_cache, dst_cache)
                    print(f"Linked cache: {cache_dir} -> {lane_name}")

            self.registry[lane_name] = {
                "branch": target_branch,
                "base": base,
                "created_at": datetime.now().isoformat()
            }
            self._save_registry()
            print(f"Lane {lane_name} successfully deployed at {target_path}")
        except subprocess.CalledProcessError as e:
            print(f"Git Error: {e.stderr}")

    def test(self, lane_name: str, cmd: str):
        lane_path = self.worktrees_dir / lane_name
        if not lane_path.exists():
            print(f"Error: Lane {lane_name} not found.")
            return 1
        
        print(f"Executing verification in {lane_name}...")
        try:
            # Run the command in the worktree directory
            process = subprocess.run(
                cmd, shell=True, cwd=lane_path, capture_output=True, text=True
            )
            print(process.stdout)
            if process.stderr:
                print(f"STDERR: {process.stderr}")
            return process.returncode
        except Exception as e:
            print(f"Execution failed: {e}")
            return 1

    def compact(self, lane_name: str, target_branch: str, message: str):
        """
        L0 -> L1 Compression: Squash all commits from lane into a single commit on target_branch.
        """
        lane_path = self.worktrees_dir / lane_name
        if not lane_path.exists():
            print(f"Error: Lane {lane_name} not found.")
            return

        print(f"Compacting L0 lane {lane_name} into L1 branch {target_branch}...")
        
        try:
            # 1. Checkout target branch in the main root to perform merge
            self._git(["checkout", target_branch])
            
            # 2. Merge with --squash to collapse L0 history
            lane_branch = self.registry[lane_name]['branch']
            self._git(["merge", "--squash", lane_branch])
            
            # 3. Commit the squashed changes
            self._git(["commit", "-m", message])
            
            print(f"Successfully compacted {lane_name} into {target_branch}.")
        except subprocess.CalledProcessError as e:
            print(f"Compaction failed: {e.stderr}")

    def project_public(self, whitelist: List[str], public_lane: str):
        """
        L1 -> L2 Projection: Sync specific files to a public-facing worktree.
        """
        public_path = self.worktrees_dir / public_lane
        if not public_path.exists():
            # Auto-spawn public lane if not exists
            self.spawn(public_lane, "private_main", branch="public_main")

        print(f"Projecting L1 -> L2 (Public) for files: {whitelist}")
        
        try:
            for file_path in whitelist:
                src_file = self.root / file_path
                dst_file = public_path / file_path
                
                if not src_file.exists():
                    print(f"Warning: {file_path} not found in source, skipping.")
                    continue
                
                # Ensure directory exists in public lane
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(src_file, dst_file)
                
                # Basic sensitivity check (L2 Guard)
                with open(dst_file, 'r', errors='ignore') as f:
                    content = f.read()
                    forbidden = ["SECRET_KEY", "PRIVATE_TOKEN", "INTERNAL_IP"]
                    for word in forbidden:
                        if word in content:
                            print(f"CRITICAL: Sensitive word {word} found in {file_path}! Aborting.")
                            return

            # Commit and push in the public lane
            self._git(["add", "."], cwd=public_path)
            # Only commit if there are changes
            status = self._git(["status", "--porcelain"], cwd=public_path)
            if status:
                self._git(["commit", "-m", f"L2 Projection: Sync {datetime.now().date()}"], cwd=public_path)
                print(f"Public lane {public_lane} updated and committed.")
            else:
                print("No changes to project.")
                
        except Exception as e:
            print(f"Projection failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Pando-Lite: Parallel Git Worktree Manager")
    subparsers = parser.add_subparsers(dest="command")

    # Status
    subparsers.add_parser("status")

    # Spawn
    spawn_parser = subparsers.add_parser("spawn")
    spawn_parser.add_argument("lane_name")
    spawn_parser.add_argument("--base", default="private_main")
    spawn_parser.add_argument("--branch", default=None)

    # Test
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("lane_name")
    test_parser.add_argument("--cmd", required=True)

    # Compact
    compact_parser = subparsers.add_parser("compact")
    compact_parser.add_argument("lane_name")
    compact_parser.add_argument("--into", required=True)
    compact_parser.add_argument("--message", required=True)

    # Project Public
    pub_parser = subparsers.add_parser("project-public")
    pub_parser.add_argument("--whitelist", required=True, help="Comma separated files")
    pub_parser.add_argument("--public-lane", default="public_main")

    args = parser.parse_args()
    manager = PandoLaneManager()

    if args.command == "status":
        manager.status()
    elif args.command == "spawn":
        manager.spawn(args.lane_name, args.base, args.branch)
    elif args.command == "test":
        exit_code = manager.test(args.lane_name, args.cmd)
        exit(exit_code)
    elif args.command == "compact":
        manager.compact(args.lane_name, args.into, args.message)
    elif args.command == "project-public":
        files = args.whitelist.split(",")
        manager.project_public(files, args.public_lane)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()