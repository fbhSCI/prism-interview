"""Run the real PowerShell installer against isolated synthetic runtimes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "installer/Install.ps1"
POWERSHELL = shutil.which("powershell.exe")
pytestmark = pytest.mark.skipif(os.name != "nt" or not POWERSHELL,
                                reason="Windows PowerShell 5.1 installer integration")


def quote(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run_ps(code: str, *, env=None):
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", code],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
        timeout=40,
    )


@pytest.fixture
def package(tmp_path):
    old = tmp_path / "old installation"
    payload = tmp_path / "installer payload"
    old.mkdir(); payload.mkdir()
    rows = []
    for relative, text in {"_internal/runtime.dll": "runtime fixture",
                           "gpu/nvidia/cublas/bin/gpu.dll": "gpu fixture",
                           "licenses/dependency.txt": "public license fixture"}.items():
        path = old / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        rows.append({"path": relative, "bytes": path.stat().st_size,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (old / "PrismInterview.exe").write_text("old executable", encoding="utf-8")
    (old / "data").mkdir()
    (old / "data/preferences.json").write_text('{"private":"fixture-user-settings"}', encoding="utf-8")
    (old / "_internal/not-in-manifest.txt").write_text("exclude me", encoding="utf-8")
    (payload / "PrismInterview.exe").write_bytes(b"new executable fixture; must not be started")
    (payload / "README.en.md").write_text("User instructions", encoding="utf-8")
    (payload / "docs").mkdir()
    (payload / "docs/guide.md").write_text("Guide", encoding="utf-8")
    manifest = tmp_path / "trusted-runtime-manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "base_version": "0.4.0", "files": rows}), encoding="utf-8")
    target = tmp_path / "new 安装 directory" / "Prism Interview 0.4.2"
    env = os.environ.copy()
    env["LOCALAPPDATA"] = str(tmp_path / "local-user-app-data")
    return dict(old=old, payload=payload, manifest=manifest, target=target, env=env, rows=rows)


def arguments(p, *, destination=True):
    args = ["-BaseDirectory " + quote(p["old"]), "-PayloadDirectory " + quote(p["payload"]),
            "-RuntimeManifest " + quote(p["manifest"]), "-NoPrompt", "-SkipShortcut"]
    if destination:
        args += ["-DestinationDirectory " + quote(p["target"])]
    return " ".join(args)


def install(p):
    return run_ps("& " + quote(SCRIPT) + " " + arguments(p), env=p["env"])


def old_hashes(p):
    return {x.relative_to(p["old"]).as_posix(): hashlib.sha256(x.read_bytes()).hexdigest()
            for x in p["old"].rglob("*") if x.is_file()}


def test_installs_new_version_only_and_keeps_old_data_unchanged(package):
    p = package
    before = old_hashes(p)
    result = install(p)
    assert result.returncode == 0, result.stdout + result.stderr
    assert old_hashes(p) == before
    assert (p["target"] / "PrismInterview.exe").read_bytes() == (p["payload"] / "PrismInterview.exe").read_bytes()
    for row in p["rows"]:
        assert hashlib.sha256((p["target"] / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    assert not (p["target"] / "data").exists()
    assert not (p["target"] / "_internal/not-in-manifest.txt").exists()
    assert json.loads((p["target"] / "runtime.json").read_text()) == {
        "data_dir": "data", "model_dir": "models", "cuda_dir": "gpu"}
    receipt_text = (p["target"] / "installation-manifest.json").read_text()
    assert str(p["old"]) not in receipt_text and str(p["payload"]) not in receipt_text
    receipt = json.loads(receipt_text)
    for record in receipt["runtime_files"] + receipt["payload_files"]:
        assert set(record) == {"path", "bytes", "sha256"}
        assert not Path(record["path"]).is_absolute()
    assert not list(p["target"].parent.glob(".prism-*"))


def test_fresh_default_directory_is_versioned_and_does_not_start_program(package):
    p = package
    result = run_ps("& " + quote(SCRIPT) + " " + arguments(p, destination=False), env=p["env"])
    assert result.returncode == 0, result.stdout + result.stderr
    target = Path(p["env"]["LOCALAPPDATA"]) / "Programs/PrismInterview/0.4.2"
    assert (target / "PrismInterview.exe").is_file()
    assert not (target / "data").exists()


def test_existing_nonempty_installation_is_never_overwritten(package):
    p = package
    p["target"].mkdir(parents=True)
    marker = p["target"] / "my-existing-record.txt"
    marker.write_text("keep this", encoding="utf-8")
    result = install(p)
    assert result.returncode != 0
    assert marker.read_text() == "keep this"
    assert sorted(x.name for x in p["target"].iterdir()) == [marker.name]
    assert not list(p["target"].parent.glob(".prism-*"))


def test_existing_empty_destination_can_be_used(package):
    p = package
    p["target"].mkdir(parents=True)
    result = install(p)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (p["target"] / "PrismInterview.exe").is_file()


def test_corrupted_base_file_fails_before_any_installation_is_staged(package):
    p = package
    (p["old"] / p["rows"][0]["path"]).write_text("corrupted source", encoding="utf-8")
    before = old_hashes(p)
    result = install(p)
    assert result.returncode != 0
    assert not p["target"].exists()
    assert old_hashes(p) == before
    assert not list(p["target"].parent.glob(".prism-*"))


@pytest.mark.parametrize("bad_path", ["../escape.dll", "data/preferences.json", "_internal/../escape.dll",
                                     "_internal/C:stream", "_internal/CON.txt", "_internal/trailing. "])
def test_manifest_rejects_unsafe_or_private_paths(package, bad_path):
    p = package
    data = json.loads(p["manifest"].read_text())
    data["files"][0]["path"] = bad_path
    p["manifest"].write_text(json.dumps(data), encoding="utf-8")
    result = install(p)
    assert result.returncode != 0
    assert not p["target"].exists()


def test_manifest_rejects_case_insensitive_duplicates(package):
    p = package
    data = json.loads(p["manifest"].read_text())
    data["files"].append(dict(data["files"][0], path="_INTERNAL/RUNTIME.DLL"))
    p["manifest"].write_text(json.dumps(data), encoding="utf-8")
    assert install(p).returncode != 0
    assert not p["target"].exists()


@pytest.mark.parametrize("relative", ["runtime.json", "data/private.txt", "source/auth.json", "source/.env"])
def test_payload_cannot_copy_runtime_settings_or_user_data(package, relative):
    p = package
    path = p["payload"] / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("do not copy", encoding="utf-8")
    assert install(p).returncode != 0
    assert not p["target"].exists()


def test_destination_nested_inside_old_program_is_rejected(package):
    p = package
    p["target"] = p["old"] / "nested-new-version"
    before = old_hashes(p)
    assert install(p).returncode != 0
    assert not p["target"].exists()
    assert old_hashes(p) == before


def test_failed_copy_retains_own_stage_without_committing_or_modifying_old(package):
    p = package
    before = old_hashes(p)
    code = ". " + quote(SCRIPT) + "; function Copy-PrismCheckedFile { throw 'simulated interrupted copy' }; "
    code += "try { $null=Invoke-PrismInstall " + arguments(p) + "; exit 0 } catch { exit 7 }"
    result = run_ps(code, env=p["env"])
    assert result.returncode == 7
    assert not p["target"].exists()
    assert len(list(p["target"].parent.glob(".prism-*"))) == 1
    assert old_hashes(p) == before


def test_destination_appearing_during_install_is_preserved(package):
    p = package
    before = old_hashes(p)
    code = ". " + quote(SCRIPT) + "; $script:originalCopy=${function:Copy-PrismCheckedFile}; "
    code += "$script:newTarget=" + quote(p["target"]) + "; "
    code += "function Copy-PrismCheckedFile { param($Source,$Destination,$Bytes,$Hash); "
    code += "if(-not(Test-Path -LiteralPath $script:newTarget)){ $null=New-Item -ItemType Directory -Path $script:newTarget -Force; "
    code += "[IO.File]::WriteAllText((Join-Path $script:newTarget 'user-file.txt'),'preserve'); }; "
    code += "& $script:originalCopy $Source $Destination $Bytes $Hash }; "
    code += "try { $null=Invoke-PrismInstall " + arguments(p) + "; exit 0 } catch { exit 7 }"
    result = run_ps(code, env=p["env"])
    assert result.returncode == 7
    assert (p["target"] / "user-file.txt").read_text() == "preserve"
    assert not (p["target"] / "PrismInterview.exe").exists()
    assert old_hashes(p) == before


@pytest.mark.parametrize("entries", [["PrismInterview/../../escape.txt"], ["/PrismInterview/x"],
                                    ["PrismInterview/a:stream"], ["OtherRoot/a.dll"],
                                    ["PrismInterview/_internal/A.dll", "PrismInterview/_internal/a.dll"]])
def test_archive_path_validation_rejects_traversal_and_aliases(tmp_path, entries):
    archive = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in entries:
            z.writestr(name, "fixture")
    code = ". " + quote(SCRIPT) + "; try { Assert-PrismArchivePaths " + quote(archive) + "; exit 0 } catch { exit 7 }"
    assert run_ps(code).returncode == 7
    assert not (tmp_path / "escape.txt").exists()


def test_archive_symlink_is_rejected(tmp_path):
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as z:
        entry = zipfile.ZipInfo("PrismInterview/_internal/linked")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(entry, "../../outside")
    code = ". " + quote(SCRIPT) + "; try { Assert-PrismArchivePaths " + quote(archive) + "; exit 0 } catch { exit 7 }"
    assert run_ps(code).returncode == 7


def test_zip_extraction_copies_only_whitelisted_runtime(package, tmp_path):
    p = package
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for row in p["rows"]:
            z.write(p["old"] / row["path"], "PrismInterview/" + row["path"])
        z.writestr("PrismInterview/data/private.txt", "not runtime")
        z.writestr("PrismInterview/_internal/unlisted.dll", "not whitelisted")
    stage = tmp_path / "archive-stage"
    stage.mkdir()
    code = ". " + quote(SCRIPT) + "; $rows=@(Read-PrismRuntimeManifest " + quote(p["manifest"]) + "); "
    code += "Install-PrismArchiveRuntime " + quote(archive) + " " + quote(stage) + " $rows"
    result = run_ps(code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {x.relative_to(stage).as_posix() for x in stage.rglob("*") if x.is_file()} == {r["path"] for r in p["rows"]}


def test_payload_junction_is_not_followed(package, tmp_path):
    p = package
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("not part of payload")
    junction = p["payload"] / "docs/linked"
    result = run_ps("New-Item -ItemType Junction -Path " + quote(junction) + " -Target " + quote(outside))
    if result.returncode != 0:
        pytest.skip("Creating test junction is unavailable on this Windows volume")
    try:
        assert install(p).returncode != 0
        assert not p["target"].exists()
        assert (outside / "private.txt").read_text() == "not part of payload"
    finally:
        if junction.exists():
            os.rmdir(junction)  # Remove only the test junction, never its contents.


def test_corrupted_cached_base_is_refused_without_overwriting_or_network(tmp_path):
    cache = tmp_path / "owned cache"
    cache.mkdir()
    archive = cache / "PrismInterview-0.4.0-Windows-x64.zip"
    archive.write_bytes(b"corrupted fixture cache")
    code = ". " + quote(SCRIPT) + "; function Invoke-WebRequest { throw 'network should not be used' }; "
    code += "try { $null=Get-PrismBaseArchive " + quote(cache) + "; exit 0 } catch { exit 7 }"
    assert run_ps(code).returncode == 7
    assert archive.read_bytes() == b"corrupted fixture cache"
    assert len(list(cache.iterdir())) == 1


def test_valid_cached_base_does_not_download_again(tmp_path):
    cache = tmp_path / "owned cache"
    cache.mkdir()
    archive = cache / "PrismInterview-0.4.0-Windows-x64.zip"
    archive.write_bytes(b"synthetic cached runtime")
    code = ". " + quote(SCRIPT) + "; $script:PrismBaseHash=" + quote(hashlib.sha256(archive.read_bytes()).hexdigest())
    code += "; $script:PrismBaseBytes=" + str(archive.stat().st_size) + "; "
    code += "function Invoke-WebRequest { throw 'network should not be used' }; "
    code += "$result=Get-PrismBaseArchive " + quote(cache) + "; if($result -ne " + quote(archive) + "){exit 7}"
    result = run_ps(code)
    assert result.returncode == 0, result.stdout + result.stderr


def test_novice_cmd_entrypoint_handles_spaces_and_unicode_paths(package):
    p = package
    wrapper = ROOT / "installer/安装PrismInterview.cmd"
    command_args = [str(wrapper), "-BaseDirectory", str(p["old"]),
                    "-PayloadDirectory", str(p["payload"]), "-RuntimeManifest", str(p["manifest"]),
                    "-DestinationDirectory", str(p["target"]), "-NoPrompt", "-SkipShortcut"]
    command = "call " + subprocess.list2cmdline(command_args)
    # cmd.exe /c interprets its tail as command text; passing that tail through
    # list2cmdline a second time would add literal backslash-escaped quotes.
    result = subprocess.run('cmd.exe /d /s /c "' + command + '"', input="\n", capture_output=True,
                            text=True, encoding="utf-8", errors="replace", env=p["env"], timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (p["target"] / "PrismInterview.exe").is_file()
    assert not (p["target"] / "data").exists()


@pytest.mark.parametrize("old_patch_missing", [False, True])
def test_reviewed_runtime_patch_replaces_only_new_installation(package, old_patch_missing):
    p = package
    old_file = p["old"] / "_internal/base_library.zip"
    old_file.write_bytes(b"old frozen standard library fixture")
    data = json.loads(p["manifest"].read_text())
    data["files"].append({"path": "_internal/base_library.zip", "bytes": old_file.stat().st_size,
                          "sha256": hashlib.sha256(old_file.read_bytes()).hexdigest()})
    p["manifest"].write_text(json.dumps(data), encoding="utf-8")
    new_file = p["payload"] / "runtime-patch/_internal/base_library.zip"
    new_file.parent.mkdir(parents=True)
    new_file.write_bytes(b"new frozen standard library fixture")
    if old_patch_missing:
        old_file.unlink()
    before = old_hashes(p)
    result = install(p)
    assert result.returncode == 0, result.stdout + result.stderr
    assert old_hashes(p) == before
    assert (p["target"] / "_internal/base_library.zip").read_bytes() == new_file.read_bytes()
    assert not (p["target"] / "runtime-patch").exists()
    receipt = json.loads((p["target"] / "installation-manifest.json").read_text())
    assert "_internal/base_library.zip" not in [row["path"] for row in receipt["runtime_files"]]
    row = next(row for row in receipt["payload_files"] if row["path"] == "_internal/base_library.zip")
    assert row["sha256"] == hashlib.sha256(new_file.read_bytes()).hexdigest()
    assert row["bytes"] == new_file.stat().st_size


@pytest.mark.parametrize("relative", ["runtime-patch/_internal/unreviewed.dll",
                                    "runtime-patch/_internal/base_library.zip.bak",
                                    "runtime-patch/data/preferences.json",
                                    "runtime-patch/_internal/auth.json",
                                    "_internal/base_library.zip"])
def test_runtime_patch_has_only_one_explicit_allowed_destination(package, relative):
    p = package
    file = p["payload"] / relative
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(b"must not copy")
    before = old_hashes(p)
    assert install(p).returncode != 0
    assert not p["target"].exists()
    assert old_hashes(p) == before


def test_runtime_patch_duplicate_destination_is_rejected(package):
    p = package
    new_file = p["payload"] / "runtime-patch/_internal/base_library.zip"
    new_file.parent.mkdir(parents=True)
    new_file.write_bytes(b"new frozen standard library fixture")
    code = ". " + quote(SCRIPT) + "; $script:realChildren=Get-Command Get-ChildItem -CommandType Cmdlet; "
    code += "$script:patchFolder=" + quote(new_file.parent) + "; "
    code += "function Get-ChildItem { param($LiteralPath,[switch]$Force); "
    code += "$items=@(& $script:realChildren -LiteralPath $LiteralPath -Force:$Force); $items; "
    code += "if($LiteralPath -eq $script:patchFolder){$items} }; "
    code += "try { $null=Get-PrismPayloadFiles " + quote(p["payload"]) + "; exit 0 } catch { exit 7 }"
    assert run_ps(code, env=p["env"]).returncode == 7
    assert not p["target"].exists()


def test_runtime_patch_junction_is_not_followed(package, tmp_path):
    p = package
    outside = tmp_path / "outside-patch"
    outside.mkdir()
    (outside / "base_library.zip").write_bytes(b"do not follow junction")
    patch = p["payload"] / "runtime-patch"
    patch.mkdir()
    junction = patch / "_internal"
    result = run_ps("New-Item -ItemType Junction -Path " + quote(junction) + " -Target " + quote(outside))
    if result.returncode != 0:
        pytest.skip("Creating test junction is unavailable on this Windows volume")
    try:
        assert install(p).returncode != 0
        assert not p["target"].exists()
        assert (outside / "base_library.zip").read_bytes() == b"do not follow junction"
    finally:
        if junction.exists():
            os.rmdir(junction)
