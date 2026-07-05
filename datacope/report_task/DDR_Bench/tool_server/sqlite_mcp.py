#!/usr/bin/env python3
"""
SQLite MCP Server for Database Exploration
SQLite database exploration server based on the official MCP Python SDK
"""

import argparse
import asyncio
import json
import logging
import sqlite3
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os

import mcp.types as types
from base_mcp_server import BaseMCPServer, setup_logging

class SQLiteMCPServer(BaseMCPServer):
    """SQLite MCP Server for database exploration"""
    
    def __init__(self, db_path: str, enable_detailed_logging: bool = True):
        self.db_path = Path(db_path)
        super().__init__("sqlite-mcp", enable_detailed_logging)
        
        self.logger.info(f"📂 Database path: {self.db_path}")
        self.logger.info(f"📊 Database size: {self._get_db_size_mb():.2f} MB")
        self.limit_default = 20
        self.limit_max = 100
        
    def _get_db_size_mb(self) -> float:
        """Get database file size in MB"""
        try:
            if self.db_path.exists():
                size_bytes = self.db_path.stat().st_size
                return size_bytes / (1024 * 1024)
        except Exception:
            pass
        return 0.0
    

    def _setup_specific_handlers(self):
        """Setup SQLite-specific MCP handlers"""
        pass  # Base class handles common handlers
    
    def _get_specific_resources(self) -> List[types.Resource]:
        """Get SQLite-specific resources"""
        return [
            types.Resource(
                uri="sqlite-mcp://schema",
                name="Database Schema",
                description="Complete database schema information",
                mimeType="application/json"
            ),
            types.Resource(
                uri="sqlite-mcp://tables",
                name="Database Tables",
                description="List of all tables in the database",
                mimeType="application/json"
            )
        ]
    
    async def _read_specific_resource(self, uri: str) -> str:
        """Read SQLite-specific resource"""
        if uri == "sqlite-mcp://schema":
            schema_info = self._get_schema_info()
            return json.dumps(schema_info, indent=2, ensure_ascii=False)
        elif uri == "sqlite-mcp://tables":
            tables = self._get_table_list()
            return json.dumps(tables, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unknown resource: {uri}")
    
    def _get_specific_tools(self) -> List[types.Tool]:
        """Get SQLite-specific tools"""
        return [
            types.Tool(
                name="execute_query",
                description="Execute read-only database queries. Support complex queries.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute"
                        },
                        "limit": {
                            "type": "integer", 
                            "description": f"Maximum number of rows to return (default: {self.limit_default}, maximum: {self.limit_max})",
                            "default": self.limit_default,
                            "maximum": self.limit_max
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="describe_table",
                description="Get the columns in the table.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to describe"
                        }
                    },
                    "required": ["table_name"]
                }
            ),
            types.Tool(
                name="get_database_info",
                description="Get general information about the database",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
        ]
    
    async def _handle_specific_tool_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle SQLite-specific tool calls"""
        if name == "execute_query":
            return self._execute_query(
                arguments["query"], 
                arguments.get("limit", self.limit_default)
            )
        elif name == "describe_table":
            return self._describe_table(arguments["table_name"])
        elif name == "get_database_info":
            return self._get_database_info()
        else:
            raise ValueError(f"Unknown tool: {name}")

    @contextmanager
    def _get_db_connection(self):
        """Get database connection context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Allow access by column name
        
        try:
            yield conn
        finally:
            conn.close()

    def _execute_query(self, query: str, limit: int = None) -> Dict[str, Any]:
        """Execute SQL query (read-only) with intelligent data size control and enhanced error handling"""
        
        # Enforce maximum limit of 100 rows
        if limit is None:
            limit = self.limit_default
        if limit > self.limit_max:
            limit = self.limit_max
        
        # Enhanced security check: allow read-only operations
        if not self._is_read_only_query(query):
            error_msg = "Only read-only queries are allowed (SELECT, PRAGMA, EXPLAIN, WITH, etc.)"
            raise ValueError(error_msg)
        
        # Smart limit enforcement for different query types
        original_query = query
        query_upper = query.strip().upper()
        limit_applied = False
        
        # Apply intelligent limits based on query type
        if query_upper.startswith('SELECT'):
            if 'LIMIT' not in query_upper:
                query = f"{query.rstrip(';')} LIMIT {limit}"
                limit_applied = True
            else:
                # Extract existing limit and cap it if too high
                import re
                limit_match = re.search(r'LIMIT\s+(\d+)', query_upper)
                if limit_match:
                    existing_limit = int(limit_match.group(1))
                    max_safe_limit = self.limit_max  # Maximum safe limit for LLM context
                    if existing_limit > max_safe_limit:
                        query = re.sub(r'LIMIT\s+\d+', f'LIMIT {max_safe_limit}', query, flags=re.IGNORECASE)
                        limit_applied = True
                        self.logger.info(f"⚠️ Reduced LIMIT from {existing_limit} to {max_safe_limit} to prevent context overflow")

        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute(query)
                
                # Handle different types of queries
                columns = []
                rows = []
                truncated_info = {"rows_truncated": False, "content_truncated": False, "truncation_reason": None}
                
                if cursor.description:  # Query returns data
                    columns = [description[0] for description in cursor.description]
                    raw_rows = cursor.fetchall()
                    
                    # Convert to compact array format with intelligent content truncation
                    rows, truncated_info = self._process_rows_with_size_control_compact(raw_rows, columns, limit)
                    
                else:  # Query doesn't return data (e.g., some PRAGMA commands)
                    rows = [["Query executed successfully"]]
                    columns = ["result"]
                
                
                # Create compact result format
                result = {
                    "cols": columns,
                    "data": rows,
                    "count": len(rows)
                }
                
                # Add basic metadata if needed
                if truncated_info["rows_truncated"] or truncated_info["content_truncated"]:
                    result["truncated"] = True
                    if truncated_info["rows_truncated"]:
                        result["original_count"] = truncated_info["original_row_count"]
                
                return result
                
        except sqlite3.OperationalError as e:
            error_msg = str(e)
            
            # Enhanced error handling with automatic table/column information
            enhanced_error = self._enhance_sql_error(error_msg, original_query)
            
            raise ValueError(enhanced_error)
            
        except Exception as e:
            raise
    
    def _enhance_sql_error(self, error_msg: str, original_query: str) -> str:
        """Enhanced error handling with automatic table/column information"""
        enhanced_error = error_msg
        
        # Check for "no such table" error
        if "no such table:" in error_msg.lower():
            try:
                db_info = self._get_database_info()
                tables = db_info.get("tables", [])
                enhanced_error = f"{error_msg}\n\nAvailable tables: {', '.join(tables)}"
            except Exception:
                # If get_database_info fails, just return original error
                pass
        
        # Check for "no such column" error
        elif "no such column:" in error_msg.lower():
            try:
                # Extract table names from the query (simple approach)
                table_names = self._extract_table_names_from_query(original_query)
                if table_names:
                    column_info = []
                    for table_name in table_names:
                        try:
                            table_desc = self._describe_table(table_name)
                            columns = [col["name"] for col in table_desc.get("columns", [])]
                            column_info.append(f"{table_name}: {', '.join(columns)}")
                        except Exception:
                            # If describe_table fails for a specific table, skip it
                            continue
                    
                    if column_info:
                        enhanced_error = f"{error_msg}\n\nAvailable columns:\n" + "\n".join(column_info)
            except Exception:
                # If column info extraction fails, just return original error
                pass
        
        return enhanced_error
    
    def _extract_table_names_from_query(self, query: str) -> List[str]:
        """Extract table names from SQL query (simple approach)"""
        table_names = []
        
        # Simple regex to find FROM and JOIN clauses
        import re
        
        # Find FROM clause
        from_match = re.search(r'\bFROM\s+(\w+)', query, re.IGNORECASE)
        if from_match:
            table_names.append(from_match.group(1))
        
        # Find JOIN clauses
        join_matches = re.findall(r'\bJOIN\s+(\w+)', query, re.IGNORECASE)
        table_names.extend(join_matches)
        
        return list(set(table_names))  # Remove duplicates
    
    def _process_rows_with_size_control_compact(self, raw_rows: List, columns: List[str], row_limit: int) -> Tuple[List[List[Any]], Dict[str, Any]]:
        """Process rows with simple size control"""
        truncated_info = {
            "rows_truncated": False,
            "content_truncated": False,
            "original_row_count": len(raw_rows)
        }
        
        # Simple row limit
        if len(raw_rows) > row_limit:
            selected_rows = raw_rows[:row_limit]
            truncated_info["rows_truncated"] = True
        else:
            selected_rows = raw_rows
        
        # Convert to array format with basic string truncation
        processed_rows = []
        MAX_FIELD_CHARS = 1000  # Basic field length limit
        
        for row_data in selected_rows:
            row_array = []
            for i in range(len(columns)):
                value = row_data[i] if i < len(row_data) else None
                
                if value is None:
                    row_array.append(None)
                else:
                    str_value = str(value)
                    if len(str_value) > MAX_FIELD_CHARS:
                        str_value = str_value[:MAX_FIELD_CHARS-3] + "..."
                        truncated_info["content_truncated"] = True
                    row_array.append(str_value)
            
            processed_rows.append(row_array)
        
        return processed_rows, truncated_info
    
    def _generate_truncation_suggestions_compact(self, original_query: str, truncated_info: Dict[str, Any]) -> List[str]:
        """Generate simple suggestions when data is truncated"""
        tips = []
        
        if truncated_info["rows_truncated"]:
            tips.append("Use WHERE clause to filter results")
        
        if truncated_info["content_truncated"]:
            tips.append("Some text fields were truncated")
        
        return tips
    
    
    def _is_read_only_query(self, query: str) -> bool:
        """Check if query is read-only (no INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, etc.)"""
        # Remove comments and normalize whitespace
        normalized_query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)  # Remove single-line comments
        normalized_query = re.sub(r'/\*.*?\*/', '', normalized_query, flags=re.DOTALL)  # Remove multi-line comments
        normalized_query = re.sub(r'\s+', ' ', normalized_query).strip().upper()
        
        # List of allowed read-only operations
        allowed_operations = [
            'SELECT',
            'WITH',      # Common Table Expressions
            'EXPLAIN',   # Query plan analysis
            'PRAGMA',    # Database configuration queries
        ]
        
        # List of forbidden write operations
        forbidden_operations = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
            'REPLACE', 'ATTACH', 'DETACH', 'VACUUM', 'REINDEX',
            'BEGIN', 'COMMIT', 'ROLLBACK', 'SAVEPOINT', 'RELEASE'
        ]
        
        # Check if query starts with allowed operations
        query_start = normalized_query.split()[0] if normalized_query else ''
        
        # First check for forbidden operations anywhere in the query
        for forbidden in forbidden_operations:
            if forbidden in normalized_query:
                return False
        
        # Then check if it starts with an allowed operation
        return query_start in allowed_operations
    

    def _describe_table(self, table_name: str) -> Dict[str, Any]:
        """Get basic table information including structure, row count, and comments"""
        with self._get_db_connection() as conn:
            # Check if table exists
            cursor = conn.execute(f"PRAGMA table_info(`{table_name}`)")
            columns = cursor.fetchall()
            
            if not columns:
                error_msg = f"Table '{table_name}' not found"
                raise ValueError(error_msg)
            
            # Get table comment
            table_comment = None
            try:
                cursor = conn.execute("SELECT comment FROM table_comments WHERE table_name = ?", (table_name,))
                result = cursor.fetchone()
                if result:
                    table_comment = result[0]
            except:
                pass  # table_comments might not exist
            
            # Build column information with comments
            column_info = []
            for col in columns:
                col_name = col[1]
                
                # Get column comment
                column_comment = None
                try:
                    cursor = conn.execute(
                        "SELECT comment FROM column_comments WHERE table_name = ? AND column_name = ?", 
                        (table_name, col_name)
                    )
                    result = cursor.fetchone()
                    if result:
                        column_comment = result[0]
                except:
                    pass  # column_comments might not exist
                
                column_info.append({
                    "name": col_name,
                    "type": col[2],
                    "not_null": bool(col[3]),
                    "default_value": col[4],
                    "primary_key": bool(col[5]),
                    "comment": column_comment
                })
            
            # Get row count
            cursor = conn.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            row_count = cursor.fetchone()[0]
            
            return {
                "table_name": table_name,
                "row_count": row_count,
                "column_count": len(column_info),
                "comment": table_comment,
                "columns": column_info
            }


    def _get_database_info(self) -> Dict[str, Any]:
        """Get general information about the database including table comments"""
        with self._get_db_connection() as conn:
            # Get all tables
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            all_tables = [row[0] for row in cursor.fetchall()]
            
            # Filter out system tables and comment/documentation tables
            tables = [table for table in all_tables 
                     if not table.startswith('sqlite_') 
                     and not table.endswith('_comments') 
                     and not table.endswith('_documentation')]
            
            # Get table comments
            table_comments = {}
            try:
                cursor = conn.execute("SELECT table_name, comment FROM table_comments")
                for row in cursor.fetchall():
                    table_comments[row[0]] = row[1]
            except:
                pass  # table_comments might not exist
            
            # Build table info with comments
            tables_info = []
            for table in tables:
                table_info = {"name": table}
                if table in table_comments:
                    table_info["comment"] = table_comments[table]
                tables_info.append(table_info)
            
            result = {
                "database_path": str(self.db_path),
                "table_count": len(tables),
                "tables": tables,
                "tables_info": tables_info,
                "filtered_out_count": len(all_tables) - len(tables)
            }
            
            return result

    def _get_schema_info(self) -> Dict[str, Any]:
        """Get basic database schema information including comments"""
        with self._get_db_connection() as conn:
            schema_info = {"tables": {}}
            
            # Get all tables
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            all_tables = [row[0] for row in cursor.fetchall()]
            
            # Filter out system tables and comment/documentation tables
            tables = [table for table in all_tables 
                     if not table.startswith('sqlite_') 
                     and not table.endswith('_comments') 
                     and not table.endswith('_documentation')]
            
            # Get table comments
            table_comments = {}
            try:
                cursor = conn.execute("SELECT table_name, comment FROM table_comments")
                for row in cursor.fetchall():
                    table_comments[row[0]] = row[1]
            except:
                pass  # table_comments might not exist
            
            # Get column comments
            column_comments = {}
            try:
                cursor = conn.execute("SELECT table_name, column_name, comment FROM column_comments")
                for row in cursor.fetchall():
                    key = f"{row[0]}.{row[1]}"
                    column_comments[key] = row[2]
            except:
                pass  # column_comments might not exist
            
            for table in tables:
                # Table structure
                cursor = conn.execute(f"PRAGMA table_info(`{table}`)")
                columns = cursor.fetchall()
                
                # Row count
                cursor = conn.execute(f"SELECT COUNT(*) FROM `{table}`")
                row_count = cursor.fetchone()[0]
                
                # Build column info with comments
                column_list = []
                for col in columns:
                    col_name = col[1]
                    key = f"{table}.{col_name}"
                    column_comment = column_comments.get(key)
                    
                    column_list.append({
                        "name": col_name,
                        "type": col[2],
                        "not_null": bool(col[3]),
                        "default_value": col[4],
                        "primary_key": bool(col[5]),
                        "comment": column_comment
                    })
                
                schema_info["tables"][table] = {
                    "columns": column_list,
                    "row_count": row_count,
                    "comment": table_comments.get(table)
                }
            
            return schema_info

    def _get_table_list(self) -> List[str]:
        """Get list of all tables (excluding system tables)"""
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            all_tables = [row[0] for row in cursor.fetchall()]
            
            # Filter out system tables and comment/documentation tables
            tables = [table for table in all_tables 
                     if not table.startswith('sqlite_') 
                     and not table.endswith('_comments') 
                     and not table.endswith('_documentation')]
            return tables


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SQLite MCP Server for Database Exploration")

    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to SQLite database file"
    )
    parser.add_argument(
        "--config",
        help="Path to config.yaml file"
    )
    parser.add_argument(
        "--scenario",
        help="Scenario name to load from config (e.g., mimic, 10k)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log file path (default: ./logs/sqlite_mcp_server_YYYYMMDD_HHMMSS.log with timestamp)"
    )
    parser.add_argument(
        "--disable-detailed-logging",
        action="store_true",
        help="Disable detailed operation logging"
    )
    
    args = parser.parse_args()
    
    # Resolve data path
    data_path = args.data_path
    
    # Try to load from config if scenario is provided and data_path is missing
    if not data_path and args.scenario:
        try:
            # Add parent directory to path to find config module
            sys.path.append(str(Path(__file__).resolve().parent.parent))
            from config import get_config
            
            config = get_config(args.config)
            scenario_config = config.get_scenario(args.scenario)
            if scenario_config.db_path:
                data_path = scenario_config.db_path
                print(f"Loaded db_path from config for scenario '{args.scenario}': {data_path}")
        except Exception as e:
            print(f"Warning: Failed to load config: {e}")
    
    # Fallback to environment variable or default
    if not data_path:
        data_path = os.getenv('MCP_DATA_PATH', "db.sqlite")

    # Setup logging with custom log directory if set
    custom_log_dir = os.getenv('CUSTOM_LOG_DIR')
    if custom_log_dir and args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"{custom_log_dir}/sqlite_mcp_server_{timestamp}.log"
    
    logger_instance = setup_logging(args.log_level, args.log_file, "sqlite_mcp")
    
    # Get the actual log file path that was used
    actual_log_file = None
    for handler in logger_instance.handlers:
        if isinstance(handler, logging.FileHandler):
            actual_log_file = handler.baseFilename
            break
    
    if not Path(data_path).exists():
        logger_instance.error(f"❌ Database file not found: {data_path}")
        return
    
    # Log startup information
    logger_instance.info("=" * 60)
    logger_instance.info("🗄️  SQLite MCP Server with Enhanced Logging")
    logger_instance.info("=" * 60)
    logger_instance.info(f"📂 Database path: {data_path}")
    logger_instance.info(f"📝 Log level: {args.log_level}")
    logger_instance.info(f"📄 Log file: {actual_log_file}")
    logger_instance.info(f"🔧 Detailed logging: {'disabled' if args.disable_detailed_logging else 'enabled'}")
    logger_instance.info(f"🆔 Process ID: {os.getpid()}")
    print(f"🆔 Process ID: {os.getpid()}")
    try:
        server = SQLiteMCPServer(
            data_path, 
            enable_detailed_logging=not args.disable_detailed_logging
        )
        await server.run()
    except KeyboardInterrupt:
        logger_instance.info("🛑 Server interrupted by user")
    except Exception as e:
        logger_instance.error(f"❌ Server failed to start: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
