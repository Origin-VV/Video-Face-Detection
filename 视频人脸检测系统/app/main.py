import sys

import torch
from PyQt5.QtWidgets import QApplication

from app.core.database import initialize_database
from app.ui.login_window import LoginWindow


def main() -> None:
    initialize_database()
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    sys.exit(app.exec_())
