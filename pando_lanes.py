#!/usr/bin/env python3
"""
pando_lanes.py - Pando-Lite Parallel Git Worktree & L0->L1->L2 Compaction Manager
=================================================================================
Co-authored by:
  - L0 Base Draft : LocalSleep-MoE (`Gemma-4-31B-IT-NVFP4 + gen_022 16-LoRA` @ 28.92 tok/s)
  - L1 Hardening  : Kuro (OpenClaw S-Tier Architect)

Core Capabilities (inspired by OpenAI Pando):
  1. Causal Branch Isolation (`spawn` / `remove`):
     Mounts parallel Git worktrees under `.worktrees/<lane_name>` sharing the single
     `.git/objects` store (0% duplicate git object disk cost) + zero-copy symlinks
     for heavy build/venv caches (`.venv_rust`, `target`, `.cargo`).
  2. Dynamic Trunk & Porcelain Discovery (`status`):
     Auto-detects the main workspace trunk branch (`private_main` or `main`) and
     inspects all live worktrees (`Ahead / Behind`, `Clean / Dirty`, `HEAD SHA`).
  3. Isolated L0 Verification (`test`):
     Runs syntax/unit test commands inside `.worktrees/<lane_name>` before compaction.
  4. LSM-Tree L0 -> L1 Compaction (`compact`):
     Auto-commits any pending L0 edits in the worktree (excluding cache symlinks)
     and squashes the entire L0 experimental history into a single clean L1 commit
     on the trunk branch.
  5. Zero-Checkout L1 -> L2 Public Projection (`project-public`):
     Mounts `main` permanently at `.worktrees/public_main` and projects whitelisted
     files from `private_main` with strict privacy keyword scanning — zero branch
     switching in the main working directory!
"""

import argparse
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional


DEFAULT_FORBIDDEN_PATTERNS = [
    "SECRET_KEY",
    "PRIVATE_TOKEN",
    "dyson_moe_bridge",
    "run_moe_bridge.sh",
    "rust_localsleep",
    "localsleep_telemetry",
    "NEURODYNAMICS_ABLATION",
]

SHARED_CACHE_DIRS = [".venv_rust", "target", ".cargo", ".venv"]


class PandoLaneManager:
    def __init__(self, root_dir: str = ".") -> None:
        self.root = pathlib.Path(root_dir).resolve()
        self.worktrees_dir = self.root / ".worktrees"
        self.worktrees_dir.mkdir(exist_ok=True)
        self.config_file = self.worktrees_dir / "pando_registry.json"
        self.registry = self._load_registry()
        self._ensure_gitignore_worktrees()

    def _ensure_gitignore_worktrees(self) -> None:
        exclude_file = self.root / ".git" / "info" / "exclude"
        if exclude_file.exists():
            txt = exclude_file.read_text(encoding="utf-8", errors="ignore")
            entries_to_add = []
            for item in [".worktrees/", "pando_lanes_v1_raw.py"]:
                if item not in txt:
                    entries_to_add.append(item)
            if entries_to_add:
                with open(exclude_file, "a", encoding="utf-8") as f:
                    f.write("\n" + "\n".join(entries_to_add) + "\n")

    def _load_registry(self) -> Dict[str, Any]:
        if self.config_file.exists():
            try:
                return json.loads(self.config_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_registry(self) -> None:
        self.config_file.write_text(
            json.dumps(self.registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _git(self, args: List[str], cwd: Optional[pathlib.Path] = None, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git"] + args,
            cwd=cwd or self.root,
            capture_output=True,
            text=True,
            check=check,
        )

    def get_trunk_branch(self) -> str:
        res = self._git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=self.root, check=False)
        return res.stdout.strip() or "main"

    def branch_exists(self, branch_name: str) -> bool:
        res = self._git(["show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"], check=False)
        return res.returncode == 0

    def _get_commit_diff(self, base: str, target: str) -> str:
        res = self._git(["rev-list", "--left-right", "--count", f"{base}...{target}"], check=False)
        if res.returncode != 0:
            return "N/A"
        parts = res.stdout.strip().split()
        if len(parts) == 2:
            behind, ahead = parts
            return f"+{ahead} / -{behind}"
        return "0 / 0"

    def status(self) -> None:
        trunk = self.get_trunk_branch()
        trunk_sha = self._git(["rev-parse", "--short", "HEAD"], cwd=self.root, check=False).stdout.strip()
        trunk_clean = "Clean" if not self._git(["status", "--porcelain"], cwd=self.root, check=False).stdout.strip() else "Dirty"

        print(f"🌳 [Trunk Root] {self.root.name} | Branch: {trunk} ({trunk_sha}) | State: {trunk_clean}")
        print("-" * 86)
        print(f"{'Lane Name':<18} | {'Branch':<24} | {'SHA':<8} | {'State':<8} | {'vs Trunk (' + trunk + ')'}")
        print("-" * 86)

        # Discover both registered lanes and any existing directories under .worktrees
        lane_names = sorted(
            set(list(self.registry.keys()) + [d.name for d in self.worktrees_dir.iterdir() if d.is_dir()])
        )
        if not lane_names:
            print("(No active parallel worktree lanes. Use `spawn <lane_name>` to create one.)")
            return

        for lane in lane_names:
            lane_path = self.worktrees_dir / lane
            if not lane_path.exists():
                continue
            br = self._git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=lane_path, check=False).stdout.strip()
            sha = self._git(["rev-parse", "--short", "HEAD"], cwd=lane_path, check=False).stdout.strip()
            dirty_out = self._git(["status", "--porcelain"], cwd=lane_path, check=False).stdout.strip()
            # Ignore symlinked cache dirs when checking dirty state
            real_dirty = [
                line for line in dirty_out.splitlines()
                if not any(line.endswith(f" {c}") or line.endswith(f" {c}/") for c in SHARED_CACHE_DIRS)
            ]
            state_str = "Clean" if not real_dirty else f"Dirty({len(real_dirty)})"
            diff_str = self._get_commit_diff(trunk, br)
            print(f"{lane:<18} | {br:<24} | {sha:<8} | {state_str:<8} | {diff_str}")

    def spawn(self, lane_name: str, base: Optional[str] = None, branch: Optional[str] = None, link_caches: bool = True) -> pathlib.Path:
        trunk = base or self.get_trunk_branch()
        target_path = self.worktrees_dir / lane_name
        target_branch = branch or f"agent/{lane_name}"

        if target_path.exists():
            print(f"ℹ️ Lane '{lane_name}' already mounted at {target_path}")
            return target_path

        try:
            if self.branch_exists(target_branch):
                print(f"🚀 Mounting existing branch '{target_branch}' into lane '{lane_name}'...")
                self._git(["worktree", "add", str(target_path), target_branch])
            else:
                print(f"🚀 Spawning new causal lane '{lane_name}' (branch '{target_branch}' based on '{trunk}')...")
                self._git(["worktree", "add", "-b", target_branch, str(target_path), trunk])

            if link_caches:
                for cache_dir in SHARED_CACHE_DIRS:
                    src_cache = self.root / cache_dir
                    dst_cache = target_path / cache_dir
                    if src_cache.exists() and src_cache.is_dir() and not dst_cache.exists():
                        os.symlink(src_cache, dst_cache)
                        print(f"   🔗 Zero-copy cache linked: {cache_dir}")

            self.registry[lane_name] = {
                "branch": target_branch,
                "base": trunk,
                "created_at": datetime.datetime.now().isoformat(),
            }
            self._save_registry()
            print(f"✅ Lane '{lane_name}' ready at: {target_path}")
            return target_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Git worktree spawn failed: {e.stderr.strip()}", file=sys.stderr)
            sys.exit(1)

    def test(self, lane_name: str, cmd: str) -> int:
        lane_path = self.worktrees_dir / lane_name
        if not lane_path.exists():
            print(f"❌ Lane '{lane_name}' not found at {lane_path}", file=sys.stderr)
            return 1
        print(f"🧪 [L0 Verification] Running in lane '{lane_name}': {cmd}")
        proc = subprocess.run(cmd, shell=True, cwd=lane_path, capture_output=True, text=True)
        if proc.stdout.strip():
            print(proc.stdout.strip())
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        print(f"🏁 [L0 Verification] EXIT_CODE: {proc.returncode}")
        return proc.returncode

    def compact(self, lane_name: str, target_branch: Optional[str], message: str) -> None:
        """
        LSM-Tree L0 -> L1 Compaction:
        Auto-commits pending L0 changes inside the lane (excluding cache symlinks),
        then squashes the lane branch into the trunk branch in a single clean commit.
        """
        lane_path = self.worktrees_dir / lane_name
        if not lane_path.exists():
            print(f"❌ Lane '{lane_name}' not found.", file=sys.stderr)
            sys.exit(1)

        trunk = target_branch or self.get_trunk_branch()
        lane_branch = self._git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=lane_path).stdout.strip()

        # 1. Remove temporary cache symlinks before staging so they are never committed
        removed_symlinks: List[str] = []
        for cache_dir in SHARED_CACHE_DIRS:
            p = lane_path / cache_dir
            if p.is_symlink():
                p.unlink()
                removed_symlinks.append(cache_dir)

        try:
            # 2. Auto-commit any uncommitted L0 edits inside the worktree
            status_out = self._git(["status", "--porcelain"], cwd=lane_path).stdout.strip()
            if status_out:
                self._git(["add", "-A"], cwd=lane_path)
                self._git(["commit", "-m", f"wip(l0-{lane_name}): auto-snapshot before L1 compaction"], cwd=lane_path)

            # 3. Ensure root is on target_branch and squash-merge the lane branch
            cur_root_br = self.get_trunk_branch()
            if cur_root_br != trunk:
                self._git(["checkout", trunk], cwd=self.root)

            self._git(["merge", "--squash", lane_branch], cwd=self.root)
            staged = self._git(["diff", "--cached", "--name-only"], cwd=self.root).stdout.strip()
            if not staged:
                print(f"ℹ️ Nothing new to compact from '{lane_name}' into '{trunk}'.")
                return

            self._git(["commit", "-m", message], cwd=self.root)
            new_sha = self._git(["rev-parse", "--short", "HEAD"], cwd=self.root).stdout.strip()
            print(f"💎 [L0 -> L1 Compacted] Squashed lane '{lane_name}' ({lane_branch}) -> '{trunk}' @ {new_sha}")
        finally:
            # Restore cache symlinks in the lane
            for cache_dir in removed_symlinks:
                src_cache = self.root / cache_dir
                dst_cache = lane_path / cache_dir
                if src_cache.exists() and not dst_cache.exists():
                    os.symlink(src_cache, dst_cache)

    def project_public(self, whitelist: List[str], public_lane: str = "public_main", public_branch: str = "main", message: str = "") -> None:
        """
        Zero-Checkout L1 -> L2 Public Projection:
        Mounts `public_branch` (`main`) inside `.worktrees/public_main`, copies whitelisted
        files from root (`private_main`), scans for private identifiers, and commits directly
        inside the public worktree without ever switching branches in the root workspace!
        """
        public_path = self.worktrees_dir / public_lane
        if not public_path.exists():
            self.spawn(public_lane, base=public_branch, branch=public_branch, link_caches=False)

        clean_files = [f.strip() for f in whitelist if f.strip()]
        print(f"🌐 [L1 -> L2 Public Projection] Target lane: {public_lane} ({public_branch}) | Files: {clean_files}")

        for rel_path in clean_files:
            src_file = self.root / rel_path
            dst_file = public_path / rel_path
            if not src_file.exists():
                print(f"⚠️ Warning: '{rel_path}' not found in root workspace, skipping.")
                continue

            content = src_file.read_text(encoding="utf-8", errors="ignore")
            for pattern in DEFAULT_FORBIDDEN_PATTERNS:
                if pattern in content:
                    print(
                        f"🚨 [PRIVACY GUARD BLOCKED] Forbidden private identifier '{pattern}' detected in '{rel_path}'! Aborting L2 projection.",
                        file=sys.stderr,
                    )
                    sys.exit(2)

            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)

        self._git(["add", "--"] + clean_files, cwd=public_path)
        diff_cached = self._git(["diff", "--cached", "--name-only"], cwd=public_path).stdout.strip()
        if not diff_cached:
            print("✅ Public lane is already 100% up-to-date with whitelisted files.")
            return

        commit_msg = message or f"feat(public-sync): L2 whitelist projection ({datetime.date.today().isoformat()})"
        self._git(["commit", "-m", commit_msg], cwd=public_path)
        pub_sha = self._git(["rev-parse", "--short", "HEAD"], cwd=public_path).stdout.strip()
        print(f"✅ [L2 Projection Complete] Committed to '{public_branch}' @ {pub_sha} (Root stayed on '{self.get_trunk_branch()}'!)")

    def remove(self, lane_name: str, delete_branch: bool = False) -> None:
        lane_path = self.worktrees_dir / lane_name
        lane_branch = None
        if lane_path.exists():
            lane_branch = self._git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=lane_path, check=False).stdout.strip()
            for cache_dir in SHARED_CACHE_DIRS:
                p = lane_path / cache_dir
                if p.is_symlink():
                    p.unlink()
            self._git(["worktree", "remove", "--force", str(lane_path)], check=False)
        self.registry.pop(lane_name, None)
        self._save_registry()
        if delete_branch and lane_branch and lane_branch not in ("main", "private_main"):
            self._git(["branch", "-D", lane_branch], check=False)
        print(f"🧹 Removed lane '{lane_name}'.")

    def spawn_code(
        self,
        lane_name: str,
        rel_file: str,
        prompt: str,
        test_cmd: str = "",
        base: Optional[str] = None,
        model: str = "localsleep-moe",
    ) -> int:
        """
        Track-A One-Shot Parallel Lane + LocalSleep-MoE Code Architect:
          1. Spawns/mounts `.worktrees/<lane_name>`
          2. Calls `:18935/v1/chat/completions` (`call_localsleep_coder.py`) to generate `rel_file`
          3. Strips markdown code fences if present & writes `.worktrees/<lane_name>/<rel_file>`
          4. Runs L0 verification (`test_cmd` or `py_compile`) inside `.worktrees/<lane_name>`
        """
        lane_path = self.spawn(lane_name, base=base)
        dst_file = lane_path / rel_file
        coder_script = pathlib.Path("/home/node/.openclaw/workspace/projects/rust_localsleep-moe/call_localsleep_coder.py")

        proc = subprocess.run(
            [
                sys.executable,
                str(coder_script),
                "--model",
                model,
                "--prompt",
                prompt,
            ],
            capture_output=True,
            text=True,
        )
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        if proc.returncode != 0:
            return proc.returncode

        raw_code = proc.stdout.strip()
        if raw_code.startswith("```"):
            lines = raw_code.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            raw_code = "\n".join(lines).strip() + "\n"

        dst_file.parent.mkdir(parents=True, exist_ok=True)
        dst_file.write_text(raw_code, encoding="utf-8")
        print(f"📝 [Track-A L0 Draft] Wrote {len(raw_code)} chars -> {dst_file}")

        verify_cmd = test_cmd or (f"python3 -m py_compile {rel_file}" if rel_file.endswith(".py") else "")
        if verify_cmd:
            return self.test(lane_name, verify_cmd)
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Pando-Lite: Parallel Git Worktree & L0->L1->L2 Compaction Manager")
    parser.add_argument("--root", default=".", help="Repository root path")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Show trunk and all parallel worktree lanes")

    sp_spawn = subparsers.add_parser("spawn", help="Mount a parallel worktree lane")
    sp_spawn.add_argument("lane_name")
    sp_spawn.add_argument("--base", default=None, help="Base branch (default: current trunk)")
    sp_spawn.add_argument("--branch", default=None, help="Branch name (default: agent/<lane_name>)")
    sp_spawn.add_argument("--no-cache-link", action="store_true", help="Do not symlink .venv_rust/target caches")

    sp_test = subparsers.add_parser("test", help="Run verification command inside a lane")
    sp_test.add_argument("lane_name")
    sp_test.add_argument("--cmd", required=True)

    sp_compact = subparsers.add_parser("compact", help="Squash-merge an L0 lane into L1 trunk")
    sp_compact.add_argument("lane_name")
    sp_compact.add_argument("--into", default=None, help="Target trunk branch (default: current trunk)")
    sp_compact.add_argument("--message", required=True, help="Clean L1 commit message")

    sp_pub = subparsers.add_parser("project-public", help="Zero-checkout L1->L2 whitelist projection")
    sp_pub.add_argument("--whitelist", required=True, help="Comma-separated file paths")
    sp_pub.add_argument("--public-lane", default="public_main")
    sp_pub.add_argument("--public-branch", default="main")
    sp_pub.add_argument("--message", default="")

    sp_rm = subparsers.add_parser("remove", help="Unmount and clean up a worktree lane")
    sp_rm.add_argument("lane_name")
    sp_rm.add_argument("--delete-branch", action="store_true")

    sp_code = subparsers.add_parser("spawn-code", help="Track-A: Spawn parallel lane + generate code via LocalSleep-MoE (:18935) + run L0 test")
    sp_code.add_argument("lane_name")
    sp_code.add_argument("--file", required=True, help="Relative file path inside the worktree lane")
    sp_code.add_argument("--prompt", required=True, help="Coding prompt for LocalSleep-MoE")
    sp_code.add_argument("--test-cmd", default="", help="Verification command inside the lane")
    sp_code.add_argument("--base", default=None)
    sp_code.add_argument("--model", default="localsleep-moe")

    args = parser.parse_args()
    mgr = PandoLaneManager(root_dir=args.root)

    if args.command == "status":
        mgr.status()
    elif args.command == "spawn":
        mgr.spawn(args.lane_name, base=args.base, branch=args.branch, link_caches=not args.no_cache_link)
    elif args.command == "test":
        sys.exit(mgr.test(args.lane_name, args.cmd))
    elif args.command == "compact":
        mgr.compact(args.lane_name, target_branch=args.into, message=args.message)
    elif args.command == "project-public":
        mgr.project_public(
            whitelist=args.whitelist.split(","),
            public_lane=args.public_lane,
            public_branch=args.public_branch,
            message=args.message,
        )
    elif args.command == "remove":
        mgr.remove(args.lane_name, delete_branch=args.delete_branch)
    elif args.command == "spawn-code":
        sys.exit(
            mgr.spawn_code(
                lane_name=args.lane_name,
                rel_file=args.file,
                prompt=args.prompt,
                test_cmd=args.test_cmd,
                base=args.base,
                model=args.model,
            )
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
