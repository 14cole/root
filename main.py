import sys


def run_gui() -> int:
    from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

    from geometry_tab import GeometryTab
    from solver_tab import SolverTab

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Solver GUI")
            tabs = QTabWidget()
            self.geometry_tab = GeometryTab()
            self.solver_tab = SolverTab(geometry_tab=self.geometry_tab)
            tabs.addTab(self.geometry_tab, "Geometry")
            tabs.addTab(self.solver_tab, "Solver")
            self.setCentralWidget(tabs)
            self.resize(1000, 600)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


def main() -> int:
    if "--headless-hardcoded" in sys.argv:
        from headless_hardcoded import main as hardcoded_main

        return hardcoded_main()

    if "--headless" in sys.argv:
        from headless_solver import main as headless_main

        argv = [arg for arg in sys.argv[1:] if arg != "--headless"]
        return headless_main(argv)

    if "--validate-physics" in sys.argv:
        from solver_physics_validation import main as validate_physics_main

        argv = [arg for arg in sys.argv[1:] if arg != "--validate-physics"]
        return validate_physics_main(argv)
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
