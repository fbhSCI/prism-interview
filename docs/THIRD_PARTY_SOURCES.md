# Shared-library sources and replacement

The Windows application is an unpacked onedir build. The Qt and FFmpeg shared libraries remain separate files under `_internal`; keep that layout. You may replace these libraries with compatible modified builds for the debugging and modifications allowed by their licenses. The application source and build script are in this repository. No license here prevents the reverse engineering needed to debug permitted library modifications.

Use `licenses/installed-dependencies.json` in the exact application archive to identify installed versions. The following are designated upstream locations for corresponding source and build information; no model weights are included in the application archive.

- **Qt / Qt for Python**: <https://download.qt.io/official_releases/qt/> and <https://download.qt.io/official_releases/QtForPython/>; source and packaging recipes: <https://code.qt.io/pyside/pyside-setup.git/>. Match the exact PySide/Qt version in the build inventory. Qt licensing: <https://doc.qt.io/qtforpython-6/licenses.html>.
- **PyAV 15.1.0**: <https://github.com/PyAV/PyAV/tree/v15.1.0>. Binary-wheel recipes and FFmpeg dependency versions: <https://github.com/PyAV/pyav-wheels>. FFmpeg source: <https://ffmpeg.org/releases/>. The inspected development wheel's `avcodec_license()` reports **LGPL version 3 or later**, using shared libraries and `--enable-version3`. Inspect the configuration of any substituted wheel before redistributing it.
- **CTranslate2**: <https://github.com/OpenNMT/CTranslate2>; **faster-whisper**: <https://github.com/SYSTRAN/faster-whisper>; **sherpa-onnx**: <https://github.com/k2-fsa/sherpa-onnx>.
- **PySide6/Qt and FFmpeg LGPL text**: <https://www.gnu.org/licenses/lgpl-3.0.html>, with referenced GPL text at <https://www.gnu.org/licenses/gpl-3.0.html>. Package-specific notices accompany the binary under `licenses/`.

The build inventory lists the build environment; not every installed package is necessarily bundled. PyMuPDF is explicitly excluded. Optional translation engines, NVIDIA runtimes, model weights and external provider CLIs retain their own distribution terms. If maintaining a public binary release, keep these source directions and notices available with that release and update version-specific links when dependencies change.
