#!/usr/bin/env python3

import json
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Set, Optional


class BaseBatchAnalyzer(ABC):
    """Base class for batch analysis with common log file management functionality"""
    
    def __init__(self, base_log_dir: str, target_ids: Optional[Set[str]] = None, overwrite: bool = False):
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        self.target_ids = target_ids
        self.overwrite = overwrite
    
    def remove_existing_log_dir(self, subdir_name: str, force: bool = False) -> None:
        """Remove existing log directory if overwrite mode is enabled"""
        if not (self.overwrite or force):
            return
            
        subdir_path = self.base_log_dir / subdir_name
        if subdir_path.exists():
            try:
                shutil.rmtree(subdir_path)
                print(f"   🗑️ Deleted existing log directory: {subdir_name}")
            except Exception as e:
                print(f"   ⚠️ Failed to delete existing log directory: {subdir_name} - {e}")
    
    def should_process_identifier(self, identifier: str) -> bool:
        """Check if identifier should be processed based on target_ids filter"""
        if self.target_ids is None:
            return True
        return identifier in self.target_ids
    
    def is_analysis_successful(self, subdir_name: str) -> bool:
        """Check if analysis was successful by combining original logic and result quality checks"""
        subdir_path = self.base_log_dir / subdir_name
        if not subdir_path.exists():
            return False
        
        # First check: Look for session_stats_*.json files (result quality check)
        session_stats_files = list(subdir_path.glob("session_stats_*.json"))
        if not session_stats_files:
            return False
        
        # Check the most recent session stats file
        session_stats_file = max(session_stats_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(session_stats_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Check if final_summary exists and is not empty
            final_summary = session_data.get('final_summary', '')
            if not final_summary or not final_summary.strip():
                return False
            
            # Additional check: ensure the session was completed
            completed = session_data.get('completed', False)
            if not completed:
                return False
            
            # Additional check: ensure the session didn't fail due to errors
            # Check if there are any error indicators in the session stats
            errors = session_data.get('errors', 0)
            if errors > 0:
                # If there are errors, check if they're critical
                # For now, we'll be lenient and only fail if there are many errors
                if errors > 10:  # Threshold for critical error count
                    return False
            
            return True
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"   ⚠️ Failed to read session stats file: {session_stats_file} - {e}")
            return False
    
    def get_failed_identifiers(self, identifiers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get list of identifiers that failed analysis (missing session_stats.json or final_summary)"""
        failed_identifiers = []
        
        # Get actual identifiers from log directory instead of using input file identifiers
        actual_identifiers = self._get_actual_identifiers_from_logs()
        
        for identifier_info in actual_identifiers:
            identifier = identifier_info.get('identifier', '')
            subdir_name = self.get_subdir_name(identifier)
            
            if not self.is_analysis_successful(subdir_name):
                failed_identifiers.append(identifier_info)
                print(f"   ❌ Detected failed analysis: {identifier}")
        
        return failed_identifiers
    
    def _get_actual_identifiers_from_logs(self) -> List[Dict[str, Any]]:
        """Get actual identifiers from log directory instead of input file"""
        actual_identifiers = []
        
        if not self.base_log_dir.exists():
            return actual_identifiers
        
        # Find all subdirectories that match the expected pattern
        for item in self.base_log_dir.iterdir():
            if item.is_dir():
                # Extract identifier from directory name
                # This is a generic approach - subclasses can override if needed
                identifier = self._extract_identifier_from_dirname(item.name)
                if identifier:
                    # Create identifier structure using subclass method
                    identifier_info = self._create_identifier_from_logs(identifier, item.name)
                    if identifier_info:
                        actual_identifiers.append(identifier_info)
        
        print(f"   📁 Found {len(actual_identifiers)} actual analysis directories in log directory")
        return actual_identifiers
    
    def _extract_identifier_from_dirname(self, dirname: str) -> Optional[str]:
        """Extract identifier from directory name. Override in subclasses if needed."""
        # Default implementation: assume directory name is "prefix_identifier"
        # This works for patterns like "patient_12345" or "user_67890"
        if '_' in dirname:
            return dirname.split('_', 1)[1]
        return dirname
    
    @abstractmethod
    def extract_identifiers(self, source_file: Path) -> List[Dict[str, Any]]:
        """Extract identifiers from source file. Must be implemented by subclasses."""
        pass
    
    def run_single_analysis(self, identifier_info: Dict[str, Any], source_file: Path, 
                          subdir_name: str, **kwargs) -> bool:
        """Run analysis for a single identifier with enhanced failure detection."""

        # Get the command to execute from subclass
        cmd, env, identifier = self._prepare_analysis_command(identifier_info, source_file, subdir_name, **kwargs)
        
        print(f"Command: {' '.join(cmd)}")
        print(f"🚀 Executing analysis for: {identifier}")
        print(f"   📁 Log directory: {self.base_log_dir / subdir_name}")
        
        # Original logic: Check subprocess execution
        subprocess_success = False
        
        try:
            result = subprocess.run(cmd, env=env, timeout=1500)  # 25-minute timeout
            
            # Log files are already generated in the subdirectory, no need to move
            
            if result.returncode == 0:
                print(f"   ✅ Process executed successfully: {identifier}")
                subprocess_success = True
            else:
                print(f"   ❌ Process execution failed: {identifier} (return code: {result.returncode})")
                subprocess_success = False
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ Execution timeout: {identifier}")
            # Log files are already generated in the subdirectory, no need to move
            subprocess_success = False
        except Exception as e:
            print(f"   ❌ Execution exception: {identifier} - {e}")
            # Log files are already generated in the subdirectory, no need to move
            subprocess_success = False
        
        # Enhanced logic: Check result quality (only if subprocess succeeded)
        if subprocess_success:
            # Wait a moment for files to be written
            time.sleep(2)
            
            # Check if the analysis actually produced good results
            if self.is_analysis_successful(subdir_name):
                print(f"   ✅ Analysis result quality check passed: {identifier}")
                return True
            else:
                print(f"   ⚠️ Process succeeded but result quality is below standard: {identifier}")
                return False
        else:
            # If subprocess failed, the analysis is definitely failed
            return False
    
    @abstractmethod
    def _prepare_analysis_command(self, identifier_info: Dict[str, Any], source_file: Path, 
                                subdir_name: str, **kwargs) -> tuple:
        """Prepare the command, environment, and identifier for analysis. Must be implemented by subclasses.
        
        Returns:
            tuple: (cmd, env, identifier) where:
                - cmd: list of command arguments
                - env: dict of environment variables
                - identifier: string identifier for logging
        """
        pass
    
    def run_batch_analysis(self, source_file: Path, max_retries: int = 2, **kwargs) -> None:
        """Run batch analysis for all identifiers with automatic retry for failed runs"""
        source_path = Path(source_file)
        max_workers = max(1, int(kwargs.get("max_workers", 1) or 1))
        if not source_path.exists():
            print(f"❌ Error: File {source_file} does not exist")
            return
        
        print("🔍 Starting batch analysis...")
        print("=" * 50)
        print(f"📄 Source file: {source_path}")
        print(f"📁 Log directory: {self.base_log_dir}")
        print(f"🔄 Max retries: {max_retries}")
        print(f"⚙️ Max workers: {max_workers}")
        print("=" * 50)
        
        # Extract all identifiers
        identifiers = self.extract_identifiers(source_path)
        if not identifiers:
            print("❌ No identifier data found")
            return

        # Filter identifiers if target_ids is specified
        if self.target_ids is not None:
            original_count = len(identifiers)
            identifiers = [id_info for id_info in identifiers 
                          if self.should_process_identifier(id_info.get('identifier', ''))]
            filtered_count = len(identifiers)
            print(f"🔍 Filtered {filtered_count} target identifiers from {original_count} identifiers")
            
            if not identifiers:
                print("❌ No matching target identifiers found")
                return

        # Skip identifiers that already have existing log directories (when not in overwrite mode)
        if not self.overwrite:
            print(f"\n🔍 Step 2: Checking existing log directories...")
            before_skip_count = len(identifiers)
            remaining_identifiers = []
            skipped_identifiers = []
            for id_info in identifiers:
                identifier_value = id_info.get('identifier', '')
                subdir_name = self.get_subdir_name(identifier_value)
                subdir_path = self.base_log_dir / subdir_name
                if subdir_path.exists():
                    skipped_identifiers.append(identifier_value)
                else:
                    remaining_identifiers.append(id_info)

            if skipped_identifiers:
                print(f"   ⏭️ Skipping {len(skipped_identifiers)} identifiers with existing log directories")
                print(f"   📁 Keeping {len(remaining_identifiers)} identifiers to be processed")
            else:
                print(f"   ✅ No existing log directories found, all {len(identifiers)} identifiers need to be processed")
            
            identifiers = remaining_identifiers
            if not identifiers:
                print("\n✅ All targets have been processed, no need to run again.")
                print("   Hint: Use --overwrite or --retry-only to re-run.")
                return

        # Ask user for confirmation
        mode_text = "Overwrite mode" if self.overwrite else "Normal mode"
        print(f"\n🔍 Step 3: Preparing to start Agent analysis...")
        print(f"   ⚠️  About to execute analysis for {len(identifiers)} identifiers ({mode_text})")
        print(f"   📁 Logs will be saved to: {self.base_log_dir}")
        print(f"   ⚙️ Max workers: {max_workers}")
        if self.target_ids:
            print(f"   🎯 Target IDs: {', '.join(sorted(self.target_ids))}")
        response = input("\nContinue? (y/N): ").strip().lower()
        
        if response not in ['y', 'yes']:
            print("❌ User cancelled operation")
            return
        
        # Execute analysis with retry logic
        print(f"\n🚀 Step 3: Starting Agent analysis...")
        self._run_analysis_with_retry(identifiers, source_path, max_retries, **kwargs)
    
    def _run_analysis_with_retry(self, identifiers: List[Dict[str, Any]], source_path: Path, 
                                 max_retries: int, **kwargs) -> None:
        """Run analysis with automatic retry for failed runs"""
        current_identifiers = identifiers.copy()
        retry_count = 0
        max_workers = max(1, int(kwargs.get("max_workers", 1) or 1))
        
        while current_identifiers and retry_count <= max_retries:
            if retry_count == 0:
                print(f"\n🚀 Starting batch analysis...")
            else:
                print(f"\n🔄 Retry #{retry_count} for failed analyses...")
            
            print("=" * 50)
            
            successful = 0
            failed = 0

            if max_workers == 1 or len(current_identifiers) == 1:
                for i, identifier_info in enumerate(current_identifiers, 1):
                    identifier = identifier_info.get('identifier', f'item_{i}')
                    subdir_name = self.get_subdir_name(identifier)
                    
                    print(f"\n[{i}/{len(current_identifiers)}] Processing identifier {identifier}")
                    
                    # Remove existing log directory if in overwrite mode or retrying
                    if self.overwrite or retry_count > 0:
                        self.remove_existing_log_dir(subdir_name, force=retry_count > 0)
                    
                    if self.run_single_analysis(identifier_info, source_path, subdir_name, **kwargs):
                        successful += 1
                    else:
                        failed += 1
                    
                    # Add brief delay to avoid too frequent requests
                    time.sleep(2)
            else:
                print(f"⚙️ Running up to {min(max_workers, len(current_identifiers))} analyses in parallel")
                with ThreadPoolExecutor(max_workers=min(max_workers, len(current_identifiers))) as executor:
                    future_to_identifier = {}
                    for i, identifier_info in enumerate(current_identifiers, 1):
                        identifier = identifier_info.get('identifier', f'item_{i}')
                        subdir_name = self.get_subdir_name(identifier)
                        print(f"\n[{i}/{len(current_identifiers)}] Queueing identifier {identifier}")
                        future = executor.submit(
                            self._run_single_analysis_task,
                            identifier_info,
                            source_path,
                            subdir_name,
                            retry_count,
                            kwargs
                        )
                        future_to_identifier[future] = identifier
                    
                    for future in as_completed(future_to_identifier):
                        identifier = future_to_identifier[future]
                        try:
                            if future.result():
                                successful += 1
                            else:
                                failed += 1
                        except Exception as e:
                            print(f"   ❌ Execution exception: {identifier} - {e}")
                            failed += 1
            
            # Check for failed analyses and prepare for retry
            if retry_count < max_retries:
                print(f"\n🔍 Checking for failed analyses...")
                failed_identifiers = self.get_failed_identifiers(current_identifiers)
                
                if failed_identifiers:
                    print(f"📊 Found {len(failed_identifiers)} failed analyses, preparing to retry...")
                    current_identifiers = failed_identifiers
                    retry_count += 1
                else:
                    print("✅ All analyses completed successfully!")
                    break
            else:
                print(f"⚠️ Reached maximum retries ({max_retries}), stopping retries")
                break
        
        # Get actual statistics from log directory
        actual_identifiers = self._get_actual_identifiers_from_logs()
        actual_failed_identifiers = self.get_failed_identifiers([])  # Empty list since we use actual logs
        actual_successful_count = len(actual_identifiers) - len(actual_failed_identifiers)
        
        # Output final summary
        print("\n" + "=" * 50)
        print("📊 Batch analysis completed!")
        print(f"✅ Success: {actual_successful_count}")
        print(f"❌ Failed: {len(actual_failed_identifiers)}")
        if len(actual_identifiers) > 0:
            success_rate = actual_successful_count / len(actual_identifiers) * 100
            print(f"📈 Success rate: {success_rate:.1f}%")
        print(f"🔄 Retries: {retry_count}")
        print(f"📁 Logs saved to: {self.base_log_dir}")
        
        if actual_failed_identifiers:
            print(f"\n❌ The following identifiers failed analysis:")
            for identifier_info in actual_failed_identifiers:
                print(f"   - {identifier_info.get('identifier', 'unknown')}")

    def _run_single_analysis_task(
        self,
        identifier_info: Dict[str, Any],
        source_path: Path,
        subdir_name: str,
        retry_count: int,
        kwargs: Dict[str, Any]
    ) -> bool:
        """Run a single analysis task for parallel execution."""
        identifier = identifier_info.get('identifier', subdir_name)
        print(f"🚀 Starting identifier {identifier}")

        if self.overwrite or retry_count > 0:
            self.remove_existing_log_dir(subdir_name, force=retry_count > 0)

        success = self.run_single_analysis(identifier_info, source_path, subdir_name, **kwargs)
        time.sleep(2)
        return success
    
    def retry_failed_analyses(self, max_retries: int = 2, **kwargs) -> None:
        """Retry only failed analyses without running the full batch"""
        max_workers = max(1, int(kwargs.get("max_workers", 1) or 1))
        print("🔍 Checking for failed analyses...")
        print("=" * 50)
        print(f"⚙️ Max workers: {max_workers}")
        
        # Get all existing subdirectories
        subdirs = [d for d in self.base_log_dir.iterdir() if d.is_dir()]
        
        if not subdirs:
            print("❌ No analysis directories found")
            return
        
        # Extract identifiers from directory names
        identifiers = []
        for subdir in subdirs:
            # Try to extract identifier from directory name
            identifier = self._extract_identifier_from_dirname(subdir.name)
            if identifier:
                # Create identifier structure using subclass method
                identifier_info = self._create_identifier_from_logs(identifier, subdir.name)
                if identifier_info:
                    identifiers.append(identifier_info)
        
        print(f"📊 Found {len(identifiers)} analysis directories")
        
        # Check for failed analyses
        failed_identifiers = self.get_failed_identifiers(identifiers)
        
        if not failed_identifiers:
            print("✅ All analyses completed successfully, no retry needed")
            return
        
        print(f"❌ Found {len(failed_identifiers)} failed analyses")
        
        # Ask user for confirmation
        response = input(f"Retry these {len(failed_identifiers)} failed analyses? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ User cancelled operation")
            return
        
        # Run retry
        self._run_analysis_with_retry(failed_identifiers, Path("."), max_retries, **kwargs)
    
    @abstractmethod
    def get_subdir_name(self, identifier: str) -> str:
        """Get subdirectory name for the identifier. Must be implemented by subclasses."""
        pass
    
    def _create_identifier_from_logs(self, identifier: str, dirname: str) -> Optional[Dict[str, Any]]:
        """Create identifier structure from log directory. Override in subclasses if needed.
        
        Args:
            identifier: The extracted identifier from directory name
            dirname: The full directory name
            
        Returns:
            Dictionary containing identifier information, or None if should be skipped
        """
        # Default implementation: create basic structure
        return {
            "identifier": identifier,
            "data": {}
        }
