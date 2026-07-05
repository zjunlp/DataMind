"""
Difficulty filter for synthetic query filtering based on various criteria.

This module provides functionality to filter synthetic queries based on
difficulty metrics such as turn count, complexity, etc.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class FilterConfig:
    """Configuration for difficulty filtering."""
    input_dir: str
    output_dir: str
    method: str = "turn_filter"
    min_turns: int = 3
    max_turns: Optional[int] = None
    preserve_structure: bool = True
    overwrite: bool = False


class DifficultyFilter:
    """
    Filter for synthetic queries based on difficulty criteria.
    
    Supports multiple filtering methods:
    - turn_filter: Filter based on number of agent turns
    - llm_filter: Filter using LLM-based difficulty assessment (TODO)
    """
    
    def __init__(self, config: FilterConfig):
        """
        Initialize difficulty filter.
        
        Args:
            config: Filter configuration
        """
        self.config = config
        
        # Validate input directory
        if not os.path.exists(self.config.input_dir):
            raise ValueError(f"Input directory does not exist: {self.config.input_dir}")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Check if output directory is empty (unless overwrite is True)
        if not self.config.overwrite and os.listdir(self.config.output_dir):
            raise ValueError(f"Output directory is not empty: {self.config.output_dir}. Use --overwrite to continue.")
    
    def filter_trajectories(self) -> Dict[str, Any]:
        """
        Filter trajectories based on the configured method.
        
        Returns:
            Dictionary containing filtering statistics and results
        """
        print(f"🔍 Starting difficulty filtering with method: {self.config.method}")
        print(f"📂 Input directory: {self.config.input_dir}")
        print(f"📁 Output directory: {self.config.output_dir}")
        
        # Find all JSON files in input directory
        json_files = list(Path(self.config.input_dir).glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in input directory: {self.config.input_dir}")
        
        print(f"📊 Found {len(json_files)} JSON files to process")
        
        # Apply the selected filtering method
        if self.config.method == "turn_filter":
            return self._apply_turn_filter(json_files)
        elif self.config.method == "llm_filter":
            return self._apply_llm_filter(json_files)
        else:
            raise ValueError(f"Unknown filtering method: {self.config.method}")
    
    def _apply_turn_filter(self, json_files: List[Path]) -> Dict[str, Any]:
        """
        Apply turn-based filtering.
        
        Args:
            json_files: List of JSON file paths to process
            
        Returns:
            Dictionary containing filtering statistics
        """
        print(f"🎯 Applying turn filter: min_turns={self.config.min_turns}")
        if self.config.max_turns:
            print(f"   max_turns={self.config.max_turns}")
        
        filtered_count = 0
        total_count = 0
        processing_errors = 0
        
        for json_file in tqdm(json_files, desc="Filtering trajectories"):
            try:
                # Load JSON data
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                total_count += 1
                
                # Extract turns count
                turns = data.get("turns", 0)
                
                if not isinstance(turns, int):
                    print(f"⚠️  Warning: Invalid turns value in {json_file.name}: {turns}")
                    processing_errors += 1
                    continue
                
                # Apply turn filter
                if self._passes_turn_filter(turns):
                    # Copy file to output directory
                    output_path = Path(self.config.output_dir) / json_file.name
                    
                    if self.config.preserve_structure:
                        # Preserve original structure
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                    else:
                        # Could modify structure here if needed
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    filtered_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing JSON file {json_file.name}: {e}")
                processing_errors += 1
            except Exception as e:
                print(f"💥 Error processing file {json_file.name}: {e}")
                processing_errors += 1
        
        # Calculate statistics
        kept_count = filtered_count
        removed_count = total_count - kept_count - processing_errors
        keep_rate = kept_count / total_count if total_count > 0 else 0.0
        
        results = {
            "method": "turn_filter",
            "config": {
                "min_turns": self.config.min_turns,
                "max_turns": self.config.max_turns,
            },
            "statistics": {
                "total_files": total_count,
                "kept_files": kept_count,
                "removed_files": removed_count,
                "processing_errors": processing_errors,
                "keep_rate": keep_rate
            },
            "input_dir": self.config.input_dir,
            "output_dir": self.config.output_dir
        }
        
        # Print summary
        print(f"\n📈 Filtering Summary:")
        print(f"   Total files processed: {total_count}")
        print(f"   Files kept: {kept_count}")
        print(f"   Files removed: {removed_count}")
        print(f"   Processing errors: {processing_errors}")
        print(f"   Keep rate: {keep_rate:.1%}")
        
        return results
    
    def _passes_turn_filter(self, turns: int) -> bool:
        """
        Check if a trajectory passes the turn filter.
        
        Args:
            turns: Number of turns in the trajectory
            
        Returns:
            True if trajectory passes the filter, False otherwise
        """
        if turns < self.config.min_turns:
            return False
        
        if self.config.max_turns and turns > self.config.max_turns:
            return False
        
        return True
    
    def _apply_llm_filter(self, json_files: List[Path]) -> Dict[str, Any]:
        """
        Apply LLM-based filtering (TODO: Not implemented).
        
        Args:
            json_files: List of JSON file paths to process
            
        Returns:
            Dictionary containing filtering statistics
        """
        # TODO: Implement LLM-based difficulty assessment
        raise NotImplementedError("LLM-based filtering is not yet implemented")


def create_difficulty_filter(
    input_dir: str,
    output_dir: str,
    method: str = "turn_filter",
    min_turns: int = 3,
    max_turns: Optional[int] = None,
    preserve_structure: bool = True,
    overwrite: bool = False
) -> DifficultyFilter:
    """
    Convenience function to create a difficulty filter.
    
    Args:
        input_dir: Input directory containing JSON files
        output_dir: Output directory for filtered files
        method: Filtering method ("turn_filter" or "llm_filter")
        min_turns: Minimum number of turns required
        max_turns: Maximum number of turns allowed (None for no limit)
        preserve_structure: Whether to preserve original JSON structure
        overwrite: Whether to overwrite existing output directory
        
    Returns:
        Configured DifficultyFilter instance
    """
    config = FilterConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        method=method,
        min_turns=min_turns,
        max_turns=max_turns,
        preserve_structure=preserve_structure,
        overwrite=overwrite
    )
    
    return DifficultyFilter(config)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Filter synthetic query trajectories by difficulty")
    
    # Required arguments
    parser.add_argument("input_dir", type=str,
                       help="Input directory containing JSON trajectory files")
    parser.add_argument("output_dir", type=str,
                       help="Output directory for filtered files")
    
    # Filtering method
    parser.add_argument("--method", type=str, default="turn_filter",
                       choices=["turn_filter", "llm_filter"],
                       help="Filtering method to use (default: turn_filter)")
    
    # Turn filter parameters
    parser.add_argument("--min-turns", type=int, default=3,
                       help="Minimum number of turns required (default: 3)")
    parser.add_argument("--max-turns", type=int, default=None,
                       help="Maximum number of turns allowed (default: no limit)")
    
    # General options
    parser.add_argument("--preserve-structure", action="store_true", default=True,
                       help="Preserve original JSON structure (default: True)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing output directory")
    
    args = parser.parse_args()
    
    print("🚀 Starting Difficulty Filter for Synthetic Trajectories")
    print(f"Method: {args.method}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    
    # Create filter
    try:
        filter_instance = create_difficulty_filter(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            method=args.method,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            preserve_structure=args.preserve_structure,
            overwrite=args.overwrite
        )
        
        # Apply filtering
        results = filter_instance.filter_trajectories()
        
        # Save results summary
        results_file = Path(args.output_dir) / "filter_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Filtering completed successfully!")
        print(f"📊 Results summary saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Error during filtering: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())