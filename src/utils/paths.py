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

        >>> get_project_path('experiments', 'unet', 'gaussian')
        PosixPath('/path/to/project/experiments/unet/gaussian')
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


def get_model_experiments_dir(model_name: str, degradation: str) -> Path:
    """
    Get the experiment directory path for a specific model and degradation type.

    Args:
        model_name: Name of the model (e.g., 'unet', 'dncnn')
        degradation: Type of degradation (e.g., 'gaussian', 'dithering')

    Returns:
        Path to experiments/[model_name]/[degradation]/
    """
    return get_project_path("experiments", model_name, degradation)


def get_raw_data_dir() -> Path:
    """Get the raw data directory path."""
    return get_project_path("data", "raw")


def get_degraded_data_dir() -> Path:
    """Get the degraded data directory path."""
    return get_project_path("data", "degraded")


def get_specific_degraded_dir(
    degradation_type: str,
    noise_sigma: float | None = None,
    bits_per_channel: int | None = None,
    dithering_type: str | None = None,
) -> Path:
    """
    Get the degraded data directory path based on degradation parameters.
    Creates a hierarchical structure to keep different degradation configs separate.

    Args:
        degradation_type: Type of degradation ('gaussian_noise', 'quantization', etc.)
        noise_sigma: Gaussian noise sigma (for gaussian_noise)
        bits_per_channel: Bits per channel (for quantization)
        dithering_type: Dithering type (for quantization)

    Returns:
        Path to degraded data with configuration-specific subdirectories

    Examples:
        Gaussian noise with sigma=100:
            data/degraded/gaussian/sigma_100/

        Quantization with random dithering at 2-bit:
            data/degraded/dithering/random/2bit/

        Quantization with Floyd-Steinberg at 4-bit:
            data/degraded/dithering/floyd_steinberg/4bit/
    """
    base_path = get_degraded_data_dir()

    if degradation_type == "gaussian_noise":
        if noise_sigma is None:
            raise ValueError("noise_sigma must be specified for gaussian_noise")
        # Format: data/degraded/gaussian/sigma_100/
        return base_path / "gaussian" / f"sigma_{int(noise_sigma)}"

    elif degradation_type == "quantization":
        if dithering_type is None or bits_per_channel is None:
            raise ValueError(
                "dithering_type and bits_per_channel must be specified for quantization"
            )
        # Format: data/degraded/dithering/random/2bit/
        return base_path / "dithering" / dithering_type / f"{bits_per_channel}bit"

    else:
        raise ValueError(f"Unknown degradation_type: {degradation_type}")


def get_processed_data_dir() -> Path:
    """Get the processed data directory path."""
    return get_project_path("data", "processed")
