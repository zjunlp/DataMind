import re
import ast
import json

ANSI_PATTERN = re.compile(r'\x1b\[[0-9;]*m')

def clean_jupyter_output(raw_output):
    if not raw_output:
        return ""
    
    if isinstance(raw_output, list):
        output_list = raw_output
    else:
        try:
            try:
                output_list = ast.literal_eval(str(raw_output))
            except (ValueError, SyntaxError):
                try:
                    output_list = json.loads(str(raw_output))
                except json.JSONDecodeError:
                    return _clean_ansi_codes(str(raw_output))
        except Exception:
            return _clean_ansi_codes(str(raw_output))
    
    if not isinstance(output_list, list):
        return output_list
    
    cleaned_outputs = []
    
    for item in output_list:
        if not isinstance(item, dict):
            cleaned_outputs.append(item)
            continue
            
        output_type = item['type']
        cleaned_item = dict(item)
        
        if output_type == 'error':
            if 'traceback' in cleaned_item and isinstance(cleaned_item['traceback'], list):
                cleaned_traceback = []
                for line in cleaned_item['traceback']:
                    original_line = str(line)
                    clean_line = _clean_ansi_codes(original_line)
                    has_ansi = ANSI_PATTERN.search(original_line) is not None
                    is_separator = clean_line.strip().startswith('---')
                    if clean_line.strip() and not (has_ansi and is_separator):
                        cleaned_traceback.append(clean_line)
                
                # Join all traceback lines and apply length limit
                full_traceback = '\n'.join(cleaned_traceback)
                if len(full_traceback) > 2000:
                    first_part = full_traceback[:1000]
                    last_part = full_traceback[-1000:]
                    full_traceback = first_part + '\n...\n' + last_part
                
                # Split back into lines for the traceback list
                cleaned_item['traceback'] = full_traceback.split('\n')
            
        elif output_type == 'stream':
            if 'text' in cleaned_item:
                clean_text = _clean_ansi_codes(cleaned_item['text'])
                # Apply length limit to stream output (similar to traceback)
                if len(clean_text) > 2000:
                    first_part = clean_text[:1000]
                    last_part = clean_text[-1000:]
                    clean_text = first_part + '\n...\n' + last_part
                cleaned_item['text'] = clean_text
        
        cleaned_outputs.append(cleaned_item)
    
    return cleaned_outputs


def _clean_ansi_codes(text: str) -> str:
    if not text:
        return ""
    return ANSI_PATTERN.sub('', text)
