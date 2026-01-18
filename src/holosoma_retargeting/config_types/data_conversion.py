"""Configuration types for data conversion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConversionConfig:
    """Configuration for data conversion.

    This follows the pattern from holosoma's config_types.
    Uses a flat structure with all conversion parameters.
    """

    input_file: str = "/home/ubuntu/DATA1/shengyin/humanoid/holosoma/src/holosoma_retargeting/demo_results/g1/object_interaction/omomo/move_chair_original.npz"
    """Path to input motion file."""

    robot: str = "g1"
    """Robot model to use. Use str to allow dynamic robot types."""

    data_format: str = "humoto"
    """Motion data format. Use str to allow dynamic data formats."""

    object_name: str = "low_chair"
    """Override object name (default depends on robot and data type)."""

    input_fps: int = 30
    """FPS of the input motion."""

    output_fps: int = 50
    """FPS of the output motion."""

    line_range: tuple[int, int] | None = None
    """Line range (start, end) for loading data (both inclusive)."""

    has_dynamic_object: bool | str = False
    """Whether the motion has a dynamic object. Can be bool or 'True'/'False' string."""

    output_name: str = "723_omniretarget"
    """Name of the output motion npz file."""

    once: bool = False
    """Run the motion once and exit."""

    use_omniretarget_data: bool = False
    """Use OmniRetarget data format."""

    headless: bool | str = False
    """Run in headless mode without initializing the viewer."""
