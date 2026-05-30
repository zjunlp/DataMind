import re
import ast
import json

def clean_jupyter_output(raw_output, max_error_length: int = 800, max_stdout_length: int = 5000) -> str:
    """
    Clean and format Jupyter kernel output to be more readable.
    
    Args:
        raw_output: Raw output list/string from Jupyter kernel
        max_error_length: Maximum length for error messages (default: 800)
    
    Returns:
        str: Cleaned and formatted output
    """
    if not raw_output:
        return ""
    
    # Handle case where raw_output is already a list
    if isinstance(raw_output, list):
        output_list = raw_output
    
    cleaned_outputs = []
    
    for item in output_list:
        if not isinstance(item, dict):
            continue
            
        output_type = item.get('type', '')
        
        if output_type == 'result':
            # Handle successful execution result
            data = item.get('data', {})
            if isinstance(data, dict):
                # Prefer text/plain over text/html
                if 'text/plain' in data:
                    content = data['text/plain']
                    cleaned_outputs.append(content)
                elif 'text/html' in data:
                    # Strip HTML tags as fallback
                    html_content = data['text/html']
                    cleaned_content = _strip_html_tags(html_content)
                    cleaned_outputs.append(cleaned_content)
                else:
                    # Use any other text content available
                    for key, value in data.items():
                        if isinstance(value, str):
                            cleaned_outputs.append(f"{key}: {value}")
                            
        elif output_type == 'error':
            # Handle error output
            error_name = item.get('name', 'Error')
            error_value = item.get('value', '')
            traceback = item.get('traceback', [])
            
            # Clean error message
            error_msg = f"{error_name}: {error_value}"
            
            # Clean and truncate traceback
            if isinstance(traceback, list):
                
                # Join and truncate if too long
                traceback_str = '\n'.join(traceback)
                if len(traceback_str) > max_error_length:
                    traceback_str = traceback_str[:max_error_length] + "\n... (truncated)"
                
                error_msg = f"{error_msg}\n{traceback_str}"
            
            cleaned_outputs.append(error_msg)
            
        elif output_type == 'stream':
            # Handle stream output (stdout, stderr)
            name = item.get('name', '')
            text = item.get('text', '')
            if text:
                cleaned_outputs.append(f"[{name}] {text}" if name else text)
    
    return '\n'.join(cleaned_outputs) if cleaned_outputs else ""


def _clean_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    if not text:
        return ""
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


def _strip_html_tags(html: str) -> str:
    """Strip HTML tags and extract plain text."""
    if not html:
        return ""
    # Remove HTML tags
    clean = re.compile('<.*?>')
    text = clean.sub('', html)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text