"""Desktop entrypoint, including a noninteractive packaged-runtime check."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import traceback


def entrypoint():
    try:
        from zoom_copilot.app import main
        return main()
    except Exception:
        details = traceback.format_exc()
        report_flag = next((flag for flag in ("--verification-report", "--self-check") if flag in sys.argv), None)
        if report_flag:
            index = sys.argv.index(report_flag)
            if index+1 < len(sys.argv):
                Path(sys.argv[index+1]).write_text(json.dumps(
                    {"passed": False, "startup_error": details}, ensure_ascii=False, indent=2), encoding="utf-8")
            return 1
        if sys.stderr is not None:
            print(details, file=sys.stderr)
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance() or QApplication([])
            QMessageBox.critical(None, "Prism 启动失败", details[-2400:])
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(entrypoint())
