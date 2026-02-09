"""Quantum Circuit Simulator - Application entry point."""

import sys
import warnings
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, qInstallMessageHandler, QtMsgType

from quantum_sim.gui.main_window import MainWindow
from quantum_sim.gui.themes.theme_manager import ThemeManager
from quantum_sim.core.config import AppConfig

# Suppress noisy Qt/Matplotlib warnings
_SUPPRESSED_QT_PATTERNS = (
    "setPointSize",
    "QFont::",
)


def _qt_message_handler(msg_type, _context, message):
    """Custom Qt message handler that suppresses known harmless warnings."""
    if msg_type == QtMsgType.QtWarningMsg:
        for pattern in _SUPPRESSED_QT_PATTERNS:
            if pattern in message:
                return
    if msg_type == QtMsgType.QtCriticalMsg or msg_type == QtMsgType.QtFatalMsg:
        print(message, file=sys.stderr)


# Filter matplotlib UserWarnings (set_ticklabels, tight_layout, etc.)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def main():
    qInstallMessageHandler(_qt_message_handler)
    app = QApplication(sys.argv)
    
    app.setApplicationName("Quantum Circuit Simulator")
    app.setOrganizationName("QuantumSIM")
    app.setApplicationVersion("1.0.0")

    config = AppConfig.load()
    theme_mgr = ThemeManager(app)
    theme_mgr.apply_theme(config.theme)

    window = MainWindow(app=app)
    window.setWindowTitle("Quantum Circuit Simulator")
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
