"""
Synthetic data generation module for DSGym.

This module provides tools for generating synthetic training data,
including trajectory generation with pass@k evaluation and query generation
through dataset exploration.
"""

from .generators import (
    TrajectoryGenerator,
    QueryGenerator,
    UnitTestGenerator,
    UnitTestGeneratorConfig,
    create_unit_test_generator,
)
from .filters import DifficultyFilter, create_difficulty_filter, QualityFilter, create_quality_filter

__version__ = "0.1.0"

__all__ = [
    "TrajectoryGenerator",
    "QueryGenerator",
    "UnitTestGenerator",
    "UnitTestGeneratorConfig",
    "create_unit_test_generator",
    "DifficultyFilter",
    "create_difficulty_filter",
    "QualityFilter",
    "create_quality_filter",
]