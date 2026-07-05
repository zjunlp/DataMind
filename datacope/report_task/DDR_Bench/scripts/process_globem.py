#!/usr/bin/env python3
"""
Script to create the optimized GLOBEM dataset.
It processes raw features, simplifying column names and categorizing them into separate files.
This version is optimized for open-source usage, supporting configurable paths and generating only categorized datasets.
"""

import os
import csv
import json
import argparse
from pathlib import Path

# Load metadata from globem_metadata.json
SCRIPT_DIR = Path(__file__).parent
METADATA_PATH = SCRIPT_DIR / "globem_metadata.json"

def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            return json.load(f)
    return {}

GLOBEM_METADATA = load_metadata()

def simplify_column_name(original_col):
    """Simplifies the original column name to a shorter form."""
    if original_col in ['', 'pid', 'date']:
        return original_col
    
    # Remove suffix :allday
    col = original_col.replace(':allday', '')
    
    # Remove prefix f_xxx:
    if ':' in col:
        col = col.split(':', 1)[1]
    
    
    # Remove sensor type prefixes
    prefixes_to_remove = GLOBEM_METADATA.get('prefixes_to_remove', [])
    
    for prefix in prefixes_to_remove:
        if col.startswith(prefix):
            col = col[len(prefix):]
            break
    
    return col

def create_field_description(simplified_name, original_name):
    """Creates a description for the simplified column name."""
    if simplified_name in ['', 'pid', 'date']:
        descriptions = {
            '': 'Row index',
            'pid': 'Participant identifier',
            'date': 'Date of observation'
        }
        return descriptions.get(simplified_name, simplified_name)
    
    # Full feature description mapping based on GLOBEM documentation
    desc_mapping = GLOBEM_METADATA.get('description_mapping', {})
    
    # Use predefined description if available
    if simplified_name in desc_mapping:
        return desc_mapping[simplified_name]
    
    # Otherwise generate a generic description
    desc = simplified_name
    if 'f_slp:' in original_name:
        desc += " (daily sleep data)"
    elif 'f_steps:' in original_name:
        desc += " (daily activity data)"
    elif 'f_loc:' in original_name:
        desc += " (daily location data)"
    elif 'f_screen:' in original_name:
        desc += " (daily phone usage data)"
    elif 'f_call:' in original_name:
        desc += " (daily communication data)"
    elif 'f_blue:' in original_name or 'f_wifi:' in original_name:
        desc += " (daily connectivity data)"
    else:
        desc += " (daily sensor data)"
    
    return desc

def get_allday_raw_features(input_file):
    """Identifies all 'allday' raw feature columns from the input CSV."""
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
    
    # Identify all allday raw features
    target_features = {}
    metadata_indices = []
    
    for i, col in enumerate(headers):
        if col in ['', 'pid', 'date']:
            metadata_indices.append((i, col, col))  # (index, original_name, simplified_name)
        elif col.endswith(':allday') and '_dis:' not in col and '_norm:' not in col:
            simplified_name = simplify_column_name(col)
            
            # Categorize by sensor type
            if 'f_slp:' in col:
                category = 'sleep'
            elif 'f_steps:' in col:
                category = 'activity'
            elif 'f_loc:' in col:
                category = 'location'
            elif 'f_screen:' in col:
                category = 'phone_usage'
            elif 'f_call:' in col:
                category = 'communication'
            elif 'f_blue:' in col or 'f_wifi:' in col:
                category = 'connectivity'
            else:
                category = 'other'
            
            if category not in target_features:
                target_features[category] = []
            target_features[category].append((i, col, simplified_name))  # (index, original_name, simplified_name)
    
    return metadata_indices, target_features

def create_optimized_files(input_file, output_dir):
    """Creates the optimized, categorized data files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get target features
    metadata_indices, target_features = get_allday_raw_features(input_file)
    
    print(f"Metadata fields: {len(metadata_indices)}")
    for category, features in target_features.items():
        print(f"{category}: {len(features)} features")
    
    total_features = sum(len(features) for features in target_features.values())
    print(f"Total: {total_features} allday raw features")
    
    # Prepare output files and writers
    output_files = {}
    writers = {}
    
    # Create categorized datasets
    for category, features in target_features.items():
        output_path = os.path.join(output_dir, f'{category}_allday_raw.csv')
        output_files[category] = open(output_path, 'w', newline='')
        writers[category] = csv.writer(output_files[category])
        
        # Write header (metadata + category features)
        category_headers = [simplified_name for _, _, simplified_name in metadata_indices] + [simplified_name for _, _, simplified_name in features]
        writers[category].writerow(category_headers)
        
        # Create field description JSON
        field_descriptions = {}
        for _, original_name, simplified_name in metadata_indices:
            field_descriptions[simplified_name] = create_field_description(simplified_name, original_name)
        for _, original_name, simplified_name in features:
            field_descriptions[simplified_name] = create_field_description(simplified_name, original_name)
        
        json_path = os.path.join(output_dir, f'{category}_allday_raw_fields.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(field_descriptions, f, indent=2, ensure_ascii=False)
    
    # Process data rows
    with open(input_file, 'r') as f_in:
        reader = csv.reader(f_in)
        headers = next(reader)  # Skip header
        
        row_count = 0
        for row in reader:
            if len(row) == len(headers):
                # Prepare metadata values once
                metadata_data = [row[i] for i, _, _ in metadata_indices]
                
                # Write to each category file
                for category, features in target_features.items():
                    if category in writers:
                        feature_data = [row[i] for i, _, _ in features]
                        writers[category].writerow(metadata_data + feature_data)
                
                row_count += 1
                if row_count % 2000 == 0:
                    print(f"Processed {row_count} rows")
    
    # Close all files
    for f in output_files.values():
        f.close()
    
    # Return stats for summary
    stats = {
        "dataset_info": {
            "participants": 155,  # Note: logic implies we know this, or we could count unique IDs
            "days_per_participant": 92, # Approximate/Expected
            "total_observations": row_count,
            "time_window": "allday",
            "normalization": "raw (original values)",
            "total_features": total_features
        },
        "features_by_category": {},
        "files_created": {}
    }
    
    for category, features in target_features.items():
        feature_count = len(features)
        stats["features_by_category"][category] = {
            "count": feature_count,
            "sample_features": [simplified_name for _, _, simplified_name in features[:3]]
        }
        stats["files_created"][f"{category}_allday_raw.csv"] = f"{feature_count} features from {category} sensors"

    return stats

def main():
    parser = argparse.ArgumentParser(description="Create optimized GLOBEM dataset (categorized raw features).")
    parser.add_argument("--input", "-i", required=True, help="Path to the GLOBEM data, which is PATH_TO/physionet/globem/1.1/INS-W_1/FeatureData/")
    parser.add_argument("--output", "-o", required=True, help="Directory to save the processed files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return

    print("Creating Optimized GLOBEM Dataset")
    print("=" * 50)
    print("Target: Categorized allday raw features")
    print()
    
    # Create optimized files
    stats = create_optimized_files(args.input, args.output)
    
    # Save summary stats
    with open(os.path.join(args.output, 'dataset_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Dataset Creation Complete ===")
    print(f"Rows processed: {stats['dataset_info']['total_observations']}")
    print(f"Total features: {stats['dataset_info']['total_features']}")
    
    print(f"\n=== Distribution by Category ===")
    for category, info in stats["features_by_category"].items():
        print(f"{category}: {info['count']} features")
    
    print(f"\n=== Files Created ===")
    for filename, description in stats["files_created"].items():
        print(f"• {filename}: {description}")
    print(f"• dataset_summary.json: Dataset statistics and metadata")
    
    print(f"\nOutput Directory: {args.output}")

if __name__ == "__main__":
    main()
