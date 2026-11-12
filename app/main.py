"""Realtime Surveillance — application entry point."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .core.settings import Settings
from .ui.main_window import MainWindow
from .utils.paths import CONFIG_PATH
from .utils.seed import set_global_seed


def main() -> int:
    settings = Settings.load(CONFIG_PATH)
    set_global_seed(settings.performance.seed)

    app = QApplication(sys.argv)
    app.setApplicationName("Realtime Surveillance")
    app.setOrganizationName("YuksekLisansTez")
    app.setStyle("Fusion")

    win = MainWindow(settings)
    win.showMaximized()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
