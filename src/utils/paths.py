"""
Path utilities for finding project root and resolving paths
"""

from pathlib import Path


def find_project_root(marker_file: str = "pyproject.toml") -> Path:
    """
    Find the project root by looking for a marker file (e.g., pyproject.toml).
    Searches upward from the current working directory.

    Args:
        marker_file: Name of the file that marks the project root

    Returns:
        Path to the project root directory

    Raises:
        FileNotFoundError: If project root cannot be found
    """
    current = Path.cwd()

    # Search upward for marker file
    for parent in [current] + list(current.parents):
        if (parent / marker_file).exists():
            return parent

    # If not found, raise error
    raise FileNotFoundError(
        f"Could not find project root. "
        f"Looking for '{marker_file}' in {current} and parent directories."
    )


def get_project_path(*parts: str, from_root: bool = True) -> Path:
    """
    Get a path relative to the project root.

    Args:
        *parts: Path components to join (e.g., 'data', 'raw', 'images')
        from_root: If True, resolve from project root. If False, use current directory.

    Returns:
        Absolute path to the requested location

    Examples:
        >>> get_project_path('data', 'raw')
        PosixPath('/path/to/project/data/raw')

        >>> get_project_path('experiments', 'results')
        PosixPath('/path/to/project/experiments/results')
    """
    if from_root:
        root = find_project_root()
        return root.joinpath(*parts)
    else:
        return Path(*parts)


# Common path helpers
def get_experiments_dir() -> Path:
    """Get the experiments directory path."""
    return get_project_path("experiments")


def get_results_dir() -> Path:
    """Get the results directory path."""
    return get_project_path("experiments", "results")


def get_raw_data_dir() -> Path:
    """Get the raw data directory path."""
    return get_project_path("data", "raw")


def get_degraded_data_dir() -> Path:
    """Get the degraded data directory path."""
    return get_project_path("data", "degraded")
