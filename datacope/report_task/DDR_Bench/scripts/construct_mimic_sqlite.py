#!/usr/bin/env python3

import os
import gzip
import shutil
import pandas as pd
import sqlite3
from sqlalchemy import create_engine, text
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import argparse
from datetime import datetime
import json
import psutil
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_metadata():
    """Load metadata from the JSON file in the same directory as the script."""
    script_dir = Path(__file__).parent.absolute()
    metadata_path = script_dir / 'mimic_metadata.json'
    
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found at {metadata_path}. Metadata injection will be skipped.")
        return None
        
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return None

# Load global configuration
METADATA = load_metadata()
if METADATA:
    TARGET_TABLES = METADATA.get("TARGET_TABLES", [])
    TABLE_COMMENTS = METADATA.get("TABLE_COMMENTS", {})
    COLUMN_COMMENTS = METADATA.get("COLUMN_COMMENTS", {})
else:
    TARGET_TABLES = []
    TABLE_COMMENTS = {}
    COLUMN_COMMENTS = {}


class OptimizedSQLiteInserter:
    """Optimized SQLite batch inserter to handle large datasets efficiently."""
    
    def __init__(self, connection, table_name: str, columns: List[str]):
        self.connection = connection
        self.table_name = table_name
        self.columns = columns
        # SQLite variable limit is typically 32766 or 250000 depending on version, 32000 is safe
        self.max_vars_per_insert = 32000 
        self.rows_per_insert = max(1, self.max_vars_per_insert // len(columns))
        logger.info(f"Table {table_name}: Inserting batches of {self.rows_per_insert} rows ({len(columns)} columns)")
    
    def insert_dataframe(self, df: pd.DataFrame, if_exists: str = 'append'):
        """Batch insert dataframe, automatically handling SQLite limits."""
        if df.empty:
            return
        
        if if_exists == 'replace':
            self._create_table(df)
        
        total_rows = len(df)
        for i in range(0, total_rows, self.rows_per_insert):
            end_idx = min(i + self.rows_per_insert, total_rows)
            batch_df = df.iloc[i:end_idx]
            self._insert_batch(batch_df)
    
    def _create_table(self, df: pd.DataFrame):
        """Create table structure based on DataFrame types."""
        self.connection.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        
        column_defs = []
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]):
                col_type = "INTEGER"
            elif pd.api.types.is_float_dtype(df[col]):
                col_type = "REAL"
            else:
                col_type = "TEXT"
            column_defs.append(f'"{col}" {col_type}')
        
        create_sql = f"CREATE TABLE {self.table_name} ({', '.join(column_defs)})"
        self.connection.execute(create_sql)
    
    def _insert_batch(self, df: pd.DataFrame):
        if df.empty:
            return
        
        placeholders = ', '.join(['?' for _ in self.columns])
        insert_sql = f"INSERT INTO {self.table_name} VALUES ({placeholders})"
        data_tuples = [tuple(row) for row in df.values]
        
        self.connection.executemany(insert_sql, data_tuples)

class MimicDataProcessor:
    def __init__(self, mimic_iv_dir: str, db_directory: str, db_name='mimic_iv.db'):
        """
        Initialize the data processor.
        
        Args:
            mimic_iv_dir: Root directory of MIMIC-IV 3.1 Dataset (containing 'icu' and 'hosp' subdirs)
            db_directory: Directory to store the SQLite database
            db_name: Name of the database file
        """
        self.db_directory = Path(db_directory)
        self.db_directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_directory / db_name
        self.progress_file = self.db_directory / 'import_progress.json'
        
        # Define schemas and mapped directories
        self.data_dirs = {
            'icu': Path(mimic_iv_dir) / 'icu',
            'hosp': Path(mimic_iv_dir) / 'hosp'
        }
        
        self.engine = None
        self.raw_connection = None
        self.max_retries = 3
        
        logger.info(f"Database will be stored at: {self.db_path.absolute()}")
    
    def check_system_resources(self, required_space_gb: float = 5.0) -> bool:
        """Check if system has enough resources."""
        try:
            disk_usage = psutil.disk_usage(self.db_directory)
            available_gb = disk_usage.free / (1024**3)
            
            if available_gb < required_space_gb:
                logger.error(f"Insufficient disk space: {required_space_gb}GB required, {available_gb:.1f}GB available")
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Resource check failed: {e}")
            return True
    
    def save_progress(self, completed_files: List[str], failed_files: List[str]):
        """Save import progress to file."""
        try:
            progress = {
                'timestamp': datetime.now().isoformat(),
                'completed_files': completed_files,
                'failed_files': failed_files
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")
    
    def load_progress(self) -> Tuple[List[str], List[str]]:
        """Load previous import progress."""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    completed = progress.get('completed_files', [])
                    failed = progress.get('failed_files', [])
                    logger.info(f"Loaded progress: {len(completed)} files completed, {len(failed)} failed")
                    return completed, failed
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")
        return [], []
        
    def connect_database(self):
        """Establish database connection."""
        try:
            connection_string = f"sqlite:///{self.db_path}"
            self.engine = create_engine(connection_string, echo=False)
            self.raw_connection = sqlite3.connect(str(self.db_path))
            logger.info(f"Connected to SQLite database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def find_gz_files(self) -> List[Tuple[str, str, str]]:
        """
        Find all .gz or .csv files in the source directories that match TARGET_TABLES.
        
        Returns:
            List of tuples: (schema_name, file_path, table_name)
        """
        files_to_process = []
        
        for schema, data_dir in self.data_dirs.items():
            if not data_dir.exists():
                logger.warning(f"Data directory not found: {data_dir}")
                continue
                
            for file in os.listdir(data_dir):
                file_path = str(data_dir / file)
                
                # Determine table name
                if file.endswith('.csv.gz'):
                    base_name = file.replace('.csv.gz', '')
                elif file.endswith('.csv'):
                    base_name = file.replace('.csv', '')
                else:
                    continue
                
                # Construct expected table name (e.g. icu_icustays)
                table_name = f"{schema}_{base_name}"
                
                if table_name in TARGET_TABLES:
                    files_to_process.append((schema, file_path, table_name))
                else:
                    logger.debug(f"Skipping undefined table: {table_name}")
        
        logger.info(f"Found {len(files_to_process)} valid data files matching target schema.")
        return files_to_process
    
    def decompress_file(self, gz_file_path: str, output_path: str) -> bool:
        """Decompress .gz file with retries."""
        for attempt in range(self.max_retries):
            try:
                if gz_file_path.endswith('.gz'):
                    with gzip.open(gz_file_path, 'rb') as f_in:
                        with open(output_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.copy2(gz_file_path, output_path)
                return True
            except Exception as e:
                logger.warning(f"Decompression failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
        return False
    
    def optimize_sqlite_settings(self):
        """Apply PRAGMA settings for performance."""
        try:
            cursor = self.raw_connection.cursor()
            optimizations = [
                "PRAGMA journal_mode = WAL",
                "PRAGMA synchronous = NORMAL", 
                "PRAGMA cache_size = 1000000",
                "PRAGMA temp_store = memory",
                "PRAGMA mmap_size = 268435456",
            ]
            for pragma in optimizations:
                cursor.execute(pragma)
            self.raw_connection.commit()
            logger.info("SQLite performance settings applied.")
        except Exception as e:
            logger.warning(f"Failed to optimize SQLite settings: {e}")
    
    def import_csv_to_db(self, csv_file_path: str, table_name: str, chunk_size: int = 10000) -> bool:
        """Import CSV content into SQLite database."""
        for attempt in range(self.max_retries):
            try:
                # Type inference
                sample_df = pd.read_csv(csv_file_path, nrows=100, low_memory=False)
                columns = list(sample_df.columns)
                inserter = OptimizedSQLiteInserter(self.raw_connection, table_name, columns)
                
                total_rows = 0
                self.raw_connection.execute("BEGIN TRANSACTION")
                
                try:
                    first_chunk = True
                    for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size, low_memory=False):
                        chunk = self.clean_dataframe(chunk)
                        
                        if_exists = 'replace' if first_chunk else 'append'
                        inserter.insert_dataframe(chunk, if_exists=if_exists)
                        first_chunk = False
                        
                        total_rows += len(chunk)
                        
                        if total_rows % 100000 == 0:
                            logger.info(f"Imported {total_rows} rows into {table_name}")
                            
                    self.raw_connection.commit()
                    logger.info(f"Successfully imported {total_rows} rows into {table_name}")
                    return True
                    
                except Exception as e:
                    self.raw_connection.rollback()
                    raise e
                        
            except Exception as e:
                logger.error(f"Import failed for {table_name}: {e}")
                # Reconnect
                self.connect_database()
        return False
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame before insertion."""
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
                df[col] = df[col].replace({'nan': None, '': None})
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].replace([float('inf'), float('-inf')], None)
        return df
    
    def inject_metadata(self):
        """Create and populate metadata tables."""
        try:
            logger.info("Injecting metadata (comments)...")
            cursor = self.raw_connection.cursor()
            
            # Create table_comments
            cursor.execute("DROP TABLE IF EXISTS table_comments")
            cursor.execute("CREATE TABLE table_comments (table_name TEXT PRIMARY KEY, comment TEXT)")
            
            for tbl, comment in TABLE_COMMENTS.items():
                cursor.execute("INSERT INTO table_comments VALUES (?, ?)", (tbl, comment))
                
            # Create column_comments
            cursor.execute("DROP TABLE IF EXISTS column_comments")
            cursor.execute("CREATE TABLE column_comments (table_name TEXT, column_name TEXT, comment TEXT, PRIMARY KEY (table_name, column_name))")
            
            for tbl, cols in COLUMN_COMMENTS.items():
                for col, comment in cols.items():
                    cursor.execute("INSERT INTO column_comments VALUES (?, ?, ?)", (tbl, col, comment))
            
            self.raw_connection.commit()
            logger.info("Metadata injection completed.")
            
        except Exception as e:
            logger.error(f"Metadata injection failed: {e}")

    def create_views(self):
        """Create documentation views."""
        try:
            logger.info("Creating documentation views...")
            cursor = self.raw_connection.cursor()
            
            views = [
                """CREATE VIEW IF NOT EXISTS table_documentation AS
                    SELECT 
                        tc.table_name,
                        tc.comment as table_description,
                        COUNT(cc.column_name) as column_count
                    FROM table_comments tc
                    LEFT JOIN column_comments cc ON tc.table_name = cc.table_name
                    GROUP BY tc.table_name, tc.comment
                    ORDER BY tc.table_name""",
                """CREATE VIEW IF NOT EXISTS column_documentation AS
                    SELECT 
                        cc.table_name,
                        cc.column_name,
                        cc.comment as column_description,
                        tc.comment as table_description
                    FROM column_comments cc
                    LEFT JOIN table_comments tc ON cc.table_name = tc.table_name
                    ORDER BY cc.table_name, cc.column_name"""
            ]
            
            for view_sql in views:
                cursor.execute(view_sql)
                
            self.raw_connection.commit()
            logger.info("Documentation views created.")
            
        except Exception as e:
            logger.error(f"View creation failed: {e}")

    def create_indexes(self):
        """Create standard indexes for performance."""
        try:
            cursor = self.raw_connection.cursor()
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_admissions_subject_id ON hosp_admissions(subject_id)",
                "CREATE INDEX IF NOT EXISTS idx_patients_subject_id ON hosp_patients(subject_id)",
                "CREATE INDEX IF NOT EXISTS idx_prescriptions_subject_id ON hosp_prescriptions(subject_id)"
            ]
            
            for idx in indexes:
                try:
                    cursor.execute(idx)
                    logger.info(f"Created index: {idx}")
                except Exception as e:
                    # Often fails if table doesn't exist, which is fine
                    logger.warning(f"Index creation failed: {e}")
                    
            self.raw_connection.commit()
        except Exception as e:
            logger.error(f"Index creation process failed: {e}")

    def process(self, chunk_size=10000, keep_decompressed=False):
        """Main processing loop."""
        if not self.check_system_resources():
            return
            
        temp_dir = Path("temp_decompressed")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            if not self.connect_database():
                return
            
            self.optimize_sqlite_settings()
            
            files = self.find_gz_files()
            if not files:
                logger.error("No matching files found.")
                return
                
            for schema, file_path, table_name in files:
                logger.info(f"Processing: {table_name}")
                
                if file_path.endswith('.gz'):
                    output_file = temp_dir / f"{table_name}.csv"
                    if not self.decompress_file(file_path, str(output_file)):
                        continue
                else:
                    output_file = Path(file_path)
                    
                if self.import_csv_to_db(str(output_file), table_name, chunk_size):
                    if not keep_decompressed and file_path.endswith('.gz'):
                         output_file.unlink()
            
            self.create_indexes()
            self.inject_metadata()
            self.create_views()
            
        finally:
            if not keep_decompressed and temp_dir.exists():
                shutil.rmtree(temp_dir)
            if self.raw_connection:
                self.raw_connection.close()

def main():
    parser = argparse.ArgumentParser(description='Open Source MIMIC-IV SQLite Constructor')
    parser.add_argument('--mimic-iv-dir', required=True, help='Path to MIMIC-IV 3.1 root directory, e.g. PATH/TO/physionet/mimiciv/3.1')
    parser.add_argument('--output-dir', default='./database', help='Output directory for database')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Batch size for insertion')
    
    args = parser.parse_args()
    
    processor = MimicDataProcessor(
        mimic_iv_dir=args.mimic_iv_dir,
        db_directory=args.output_dir
    )
    
    processor.process(chunk_size=args.chunk_size)

if __name__ == "__main__":
    main()
