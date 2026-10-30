"""Fail closed on accidental runtime/private content in the public source tree.

This is a release hygiene check, not a complete secret detection service.
Print paths and rule names only; never print matching content.
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORE = {'.git', '.venv', '__pycache__', '.pytest_cache', 'outputs', 'dist', 'build', 'work'}
RUNTIME_PARTS = {'sessions', 'data', 'models', 'gpu', 'runtime', 'documents', 'test_media'}
FORBIDDEN = {'auth.json', 'preferences.json', 'knowledge.json', 'research_facts.json',
             'prepared_answers.json', 'portable_documents.json', 'session_api_keys.json',
             'runtime.json'}
RULES = {
    'API credential': re.compile(r'\bsk-[A-Za-z0-9_-]{24,}\b'),
    'GitHub token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'user-specific Windows path': re.compile(r'[A-Za-z]:[/\\]Users[/\\][^/\\\r\n]+', re.I),
    'private endpoint': re.compile(r'maas\.' + r'qianwenaiapi\.com', re.I),
}


def main():
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True)
    if tracked.returncode == 0 and tracked.stdout:
        paths = [ROOT / name for name in tracked.stdout.decode('utf-8').split('\0') if name]
    else:
        paths = [p for p in ROOT.rglob('*') if p.is_file() and not any(x in IGNORE for x in p.relative_to(ROOT).parts)]
    errors = []
    for path in paths:
        relative = path.relative_to(ROOT)
        if any(part in IGNORE for part in relative.parts):
            errors.append((relative, 'ignored runtime content tracked'))
            continue
        if (any(part in RUNTIME_PARTS or part.startswith('data-') for part in relative.parts)
                or path.is_symlink() or path.name in FORBIDDEN
                or path.suffix.lower() in {'.wav', '.mp3', '.mp4', '.zip', '.7z', '.rar', '.pem', '.key'}
                or path.name.startswith('.env')):
            errors.append((relative, 'private/runtime artifact'))
            continue
        if path.stat().st_size > 5_000_000:
            errors.append((relative, 'unexpected large source artifact'))
        try:
            contents = path.read_text(encoding='utf-8-sig')
        except (UnicodeError, OSError):
            continue
        for name, pattern in RULES.items():
            if pattern.search(contents):
                errors.append((relative, name))
    for path, reason in errors:
        print(f'{path.as_posix()}: {reason}')
    print(f'Public tree: {len(paths)} files checked, {len(errors)} findings')
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
