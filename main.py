import sys

from app.cli import build_parser, gui_only_misuse, run_headless


def main() -> int:
    # parse_known_args: anything argparse does not recognise is left for Qt,
    # which has flags of its own (-platform, -style, ...).
    args, qt_args = build_parser().parse_known_args()
    if args.headless or args.list:
        return run_headless(args)

    misuse = gui_only_misuse(args)
    if misuse:
        print(f"Note: {misuse}", file=sys.stderr)

    # Imported here so a headless run never loads the widget stack.
    from PyQt6.QtWidgets import QApplication

    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv[:1] + qt_args)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
