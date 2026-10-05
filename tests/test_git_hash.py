"""
tests/test_run_logger.py
Verification suite for the hardened run logger.
"""

import subprocess
from pathlib import Path
import pytest
import common.run_logger as run_logger
from common.run_logger import get_git_commit_hash


def _run_git(cwd: Path, *args: str) -> str:
    """Helper to run git commands with a throwaway identity inside a temporary directory."""
    cmd = [
        "git",
        "-c", "user.name=TestRunner",
        "-c", "user.email=test@reproducibility.org",
        *args,
    ]
    res = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


@pytest.fixture
def temp_git_repo(tmp_path: Path):
    """
    Creates a real isolated Git repository on disk with:
    - Initialized git tree
    - Tracked code.py and results.csv
    - One committed baseline commit
    """
    repo_dir = tmp_path / "sandbox_repo"
    repo_dir.mkdir()

    # 1. Initialize repo
    _run_git(repo_dir, "init")

    # 2. Create tracked files
    code_file = repo_dir / "code.py"
    code_file.write_text("print('hello baseline')\n", encoding="utf-8")

    results_file = repo_dir / "results.csv"
    results_file.write_text("timestamp,project,metric\n", encoding="utf-8")

    # 3. Commit initial state
    _run_git(repo_dir, "add", "code.py", "results.csv")
    _run_git(repo_dir, "commit", "-m", "initial baseline commit")

    return repo_dir


def test_git_commit_hash_real_repo_clean_state(temp_git_repo: Path, monkeypatch):
    monkeypatch.setattr(run_logger, "REPO_ROOT", temp_git_repo)

    commit_hash = get_git_commit_hash()
    assert commit_hash != "unknown"
    assert not commit_hash.endswith("-dirty")
    assert len(commit_hash) >= 4


def test_git_commit_hash_excludes_results_csv_modification(temp_git_repo: Path, monkeypatch):
    """
    Core regression test: Modifying results.csv MUST NOT stamp the run as -dirty.
    """
    monkeypatch.setattr(run_logger, "REPO_ROOT", temp_git_repo)

    # Modify results.csv (simulate logging an experiment run)
    results_file = temp_git_repo / "results.csv"
    results_file.write_text("timestamp,project,metric\n2026-10-04,p1,10.5\n", encoding="utf-8")

    commit_hash = get_git_commit_hash()
    assert not commit_hash.endswith("-dirty"), "results.csv modification erroneously marked tree dirty!"


def test_git_commit_hash_detects_dirty_tracked_code(temp_git_repo: Path, monkeypatch):
    """
    Modifying tracked source code MUST mark the run as -dirty.
    """
    monkeypatch.setattr(run_logger, "REPO_ROOT", temp_git_repo)

    code_file = temp_git_repo / "code.py"
    code_file.write_text("print('modified logic')\n", encoding="utf-8")

    commit_hash = get_git_commit_hash()
    assert commit_hash.endswith("-dirty"), "Modified tracked code was not detected as dirty!"


def test_git_commit_hash_ignores_untracked_scratch_files(temp_git_repo: Path, monkeypatch):
    """
    Verifies that --untracked-files=no prevents scratch files from falsely dirtying runs.
    """
    monkeypatch.setattr(run_logger, "REPO_ROOT", temp_git_repo)

    scratch = temp_git_repo / "scratch_notes.txt"
    scratch.write_text("untracked notes", encoding="utf-8")

    commit_hash = get_git_commit_hash()
    assert not commit_hash.endswith("-dirty")


def test_git_commit_hash_empty_repo_or_missing_git(tmp_path: Path, monkeypatch):
    """
    A directory with git init but zero commits returns 'unknown' gracefully.
    """
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()
    _run_git(empty_repo, "init")

    monkeypatch.setattr(run_logger, "REPO_ROOT", empty_repo)
    assert get_git_commit_hash() == "unknown"

    # Non-git directory
    non_git_dir = tmp_path / "not_a_repo"
    non_git_dir.mkdir()
    monkeypatch.setattr(run_logger, "REPO_ROOT", non_git_dir)
    assert get_git_commit_hash() == "unknown"