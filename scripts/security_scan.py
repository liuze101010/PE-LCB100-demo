from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py",
    ".js",
    ".css",
    ".html",
    ".json",
    ".md",
    ".ps1",
    ".txt",
    ".yml",
    ".yaml",
    ".sh",
    ".example",
}
TEXT_NAMES = {"Dockerfile", "Caddyfile"}
SKIP_DIRS = {
    ".git",
    ".venv-platform",
    ".venv",
    ".runtime",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
    "tmp",
}

PATTERNS = {
    "api_key": re.compile(r"(?i)\b(?:sk|key)-[a-z0-9]{20,}\b"),
    "provider_key": re.compile(r"(?i)\b[a-f0-9]{32}\.[A-Za-z0-9]{12,}\b"),
    "windows_user_path": re.compile(r"(?i)[A-Z]:\\Users\\[^\\\s]+"),
    "project_absolute_path": re.compile(r"(?i)D:\\lz project"),
    "remote_home": re.compile(r"/public/home/[A-Za-z0-9._-]+/"),
    "private_ipv4": re.compile(r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)\d{1,3}\.\d{1,3}\b"),
    "private_key": re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
}


def scan() -> list[str]:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file() or (path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_NAMES):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}: {name}")
    return findings


if __name__ == "__main__":
    issues = scan()
    if issues:
        print("\n".join(issues))
        raise SystemExit(1)
    print("Security scan passed: no keys, usernames, or private absolute paths.")
