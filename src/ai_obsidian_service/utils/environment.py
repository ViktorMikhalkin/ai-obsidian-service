# src/ai_obsidian_service/core/environment.py
from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass
class EnvironmentCheck:
    name: str
    passed: bool
    message: str
    critical: bool = True


class EnvironmentValidator:
    """Validates runtime environment for known compatibility issues."""

    def __init__(self):
        self.checks: list[EnvironmentCheck] = []

    def validate_all(self) -> bool:
        """Run all environment checks. Returns True if environment is safe."""
        self.checks.clear()

        self._check_python_version()
        self._check_faiss_compatibility()
        self._check_numpy_compatibility()
        self._check_torch_compatibility()
        self._check_platform_compatibility()

        # Report results
        critical_failures = [c for c in self.checks if not c.passed and c.critical]
        warnings_count = len([c for c in self.checks if not c.passed and not c.critical])

        if critical_failures:
            print("❌ Critical environment issues detected:")
            for check in critical_failures:
                print(f"  - {check.name}: {check.message}")
            return False

        if warnings_count > 0:
            print(f"⚠️  {warnings_count} non-critical warnings detected:")
            for check in [c for c in self.checks if not c.passed and not c.critical]:
                print(f"  - {check.name}: {check.message}")

        passed_count = len([c for c in self.checks if c.passed])
        print(f"✅ Environment validation passed ({passed_count}/{len(self.checks)} checks)")
        return True

    def _check_python_version(self) -> None:
        """Check Python version compatibility."""
        import sys
        version = sys.version_info

        if version >= (3, 12):
            self.checks.append(EnvironmentCheck(
                name="Python Version",
                passed=False,
                message=f"Python {version.major}.{version.minor} may have compatibility issues with FAISS. Consider using Python 3.11.",
                critical=False
            ))
        elif version < (3, 10):
            self.checks.append(EnvironmentCheck(
                name="Python Version",
                passed=False,
                message=f"Python {version.major}.{version.minor} is too old. Minimum required: 3.10",
                critical=True
            ))
        else:
            self.checks.append(EnvironmentCheck(
                name="Python Version",
                passed=True,
                message=f"Python {version.major}.{version.minor} is compatible"
            ))

    def _check_faiss_compatibility(self) -> None:
        """Check FAISS version and basic functionality."""
        try:
            import faiss
            version = faiss.__version__

            # Test basic functionality
            import numpy as np
            test_index = faiss.IndexFlatIP(3)
            test_data = np.array([[1, 0, 0]], dtype=np.float32)
            test_index.add(test_data)
            _, _ = test_index.search(test_data, 1)

            # Check for known problematic versions
            if version.startswith("1.9."):
                self.checks.append(EnvironmentCheck(
                    name="FAISS Version",
                    passed=False,
                    message=f"FAISS {version} has known segmentation fault issues. Use 1.8.0.",
                    critical=True
                ))
            else:
                self.checks.append(EnvironmentCheck(
                    name="FAISS Version",
                    passed=True,
                    message=f"FAISS {version} is working correctly"
                ))

        except ImportError:
            self.checks.append(EnvironmentCheck(
                name="FAISS Availability",
                passed=False,
                message="FAISS not installed. Install faiss-cpu>=1.8.0,<1.9.0",
                critical=True
            ))
        except Exception as e:
            self.checks.append(EnvironmentCheck(
                name="FAISS Functionality",
                passed=False,
                message=f"FAISS test failed: {e}",
                critical=True
            ))

    def _check_numpy_compatibility(self) -> None:
        """Check numpy version compatibility."""
        try:
            import numpy as np
            version = np.__version__

            if version.startswith("2."):
                self.checks.append(EnvironmentCheck(
                    name="NumPy Version",
                    passed=False,
                    message=f"NumPy {version} (v2.x) may have compatibility issues. Use 1.24-1.26.",
                    critical=False
                ))
            else:
                self.checks.append(EnvironmentCheck(
                    name="NumPy Version",
                    passed=True,
                    message=f"NumPy {version} is compatible"
                ))

        except ImportError:
            self.checks.append(EnvironmentCheck(
                name="NumPy Availability",
                passed=False,
                message="NumPy not available",
                critical=True
            ))

    def _check_torch_compatibility(self) -> None:
        """Check PyTorch compatibility."""
        try:
            import torch
            version = torch.__version__

            self.checks.append(EnvironmentCheck(
                name="PyTorch Version",
                passed=True,
                message=f"PyTorch {version} is available"
            ))

        except ImportError:
            self.checks.append(EnvironmentCheck(
                name="PyTorch Availability",
                passed=False,
                message="PyTorch not available",
                critical=True
            ))

    def _check_platform_compatibility(self) -> None:
        """Check platform-specific issues."""
        system = platform.system()
        machine = platform.machine()

        if system == "Linux" and machine == "x86_64":
            # Known good platform
            self.checks.append(EnvironmentCheck(
                name="Platform Compatibility",
                passed=True,
                message=f"{system} {machine} is well supported"
            ))
        else:
            self.checks.append(EnvironmentCheck(
                name="Platform Compatibility",
                passed=True,
                message=f"{system} {machine} - compatibility not verified",
                critical=False
            ))


def validate_environment() -> bool:
    """Convenience function to validate environment."""
    validator = EnvironmentValidator()
    return validator.validate_all()


def require_environment() -> None:
    """Validate environment and exit if critical issues found."""
    if not validate_environment():
        import sys
        print("\n🛑 Environment validation failed. Please fix critical issues before proceeding.")
        sys.exit(1)


# Auto-run validation on import for critical paths
def _auto_validate():
    """Run validation automatically in certain contexts."""
    import os

    if os.getenv("AIOBS_SKIP_ENV_CHECK") == "1":
        return

    # Only validate in production/test contexts
    if os.getenv("AIOBS_ENV") in ("production", "staging"):
        require_environment()
    elif os.getenv("PYTEST_CURRENT_TEST"):  # Running tests
        validate_environment()  # Warn but don't exit


# Run auto-validation
if __name__ != "__main__":
    _auto_validate()


if __name__ == "__main__":
    # CLI usage: python -m ai_obsidian_service.core.environment
    import sys
    sys.exit(0 if validate_environment() else 1)
