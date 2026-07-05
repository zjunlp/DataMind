#!/usr/bin/env python3
"""
Generate Docker Compose file for start_index to end_index containers (simplified)
"""

import yaml
import json
import argparse
import os
from typing import Optional

def generate_container_config(container_id: int, 
                              host_port: int, 
                              container_type: str = 'default', 
                              mountable_volumes: Optional[str] = None, 
                              mountable_volume_saveto: Optional[str] = None,
                              extra_env: Optional[dict] = None, 
                              manager_url: Optional[str] = None,
                              gpu_id: Optional[int] = None) -> dict:

    # default values. these can be set with the env vars passed in
    execution_timeout = extra_env['EXECUTION_TIMEOUT'] if extra_env and 'EXECUTION_TIMEOUT' in extra_env else '360'
    mem_limit = extra_env['MEM_LIMIT'] if extra_env and 'MEM_LIMIT' in extra_env else '2G'
    cpus = float(extra_env['CPUS']) if extra_env and 'CPUS' in extra_env else 0.5
    mem_reservation = extra_env['MEM_RESERVATION'] if extra_env and 'MEM_RESERVATION' in extra_env else '512M'
    
    base_env = [
        'PORT=8432',
        f'CONTAINER_ID={container_id}',
        f'HOST_PORT={host_port}',
        'PYTHONUNBUFFERED=1',
        'PYTHONDONTWRITEBYTECODE=1',
        f'EXECUTION_TIMEOUT={execution_timeout}'
    ]
    
    if extra_env:
        for key, value in extra_env.items():
            if key not in ['EXECUTION_TIMEOUT', 'MEM_LIMIT', 'CPUS', 'MEM_RESERVATION']:
                base_env.append(f'{key}={value}')
    
    config = {
        'image': container_type,
        'command': 'python main.py',
        'ports': [f'{host_port}:8432'],
        'environment': base_env,
        'restart': 'unless-stopped',
        'labels': [
            f'container.id={container_id}',
            f'container.port={host_port}',
            f'container.type={container_type}'
        ],
        'mem_limit': mem_limit,
        'cpus': cpus,
        'mem_reservation': mem_reservation,
        'extra_hosts': ['host.docker.internal:host-gateway']
    }
    
    # Add GPU configuration if specified
    if gpu_id is not None:
        config.pop('mem_reservation', None)
        config['deploy'] = {
            'resources': {
                'reservations': {
                    'memory': '128M',
                    'devices': [{
                        'driver': 'nvidia',
                        'device_ids': [str(gpu_id)],
                        'capabilities': ['gpu']
                    }]
                }
            }
        }
    
    # Set up mountable volumes
    volumes = []
    if mountable_volumes:
        volumes.append(f'{mountable_volumes}:/data:ro')
        # Set environment variable to point to mounted workspace
        base_env.append('WORKSPACE_FOLDER=/data')
    
    # Set up save directory volume
    if mountable_volume_saveto:
        container_save_dir = f'{mountable_volume_saveto}/container_{container_id:03d}'
        volumes.append(f'{container_save_dir}:/submission')
        # Set environment variable to point to submission directory
        base_env.append('SUBMISSION_DIR=/submission')
    
    if volumes:
        config['volumes'] = volumes
    
    return config

def generate_container_config_json(
    num_containers: int,
    container_assignments: dict,
    output_file: str = 'container_config.json'
) -> None:
    """Generate container_config.json for the manager"""
    
    config = {}
    for i in range(num_containers):
        container_id = str(i)
        service_name = f'executor-{i:03d}'
        container_type = container_assignments[i]
        
        config[container_id] = {
            "url": f"http://{service_name}:8432",
            "type": container_type
        }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Generated {output_file}")
    
    # Show type summary
    if container_assignments:
        type_counts = {}
        for container_type in container_assignments.values():
            type_counts[container_type] = type_counts.get(container_type, 0) + 1
        
        print("📊 Container types in config:")
        for type_name, count in type_counts.items():
            print(f"   {type_name}: {count} containers")

def generate_compose_file(
    num_containers: int,
    container_assignments: dict,
    start_port: int = 60000,
    output_file: str = 'docker-compose.yml',
    mountable_volumes: Optional[str] = None,
    mountable_volume_saveto: Optional[str] = None,
    extra_env: Optional[dict] = None,
    gpu_ids: Optional[list[int]] = None,
    config_file: str = 'container_config.json'
) -> None:
    """Generate a complete Docker Compose file"""
    
    # Validate GPU IDs if provided
    if gpu_ids is not None:
        assert len(gpu_ids) == num_containers, f"Number of GPU IDs ({len(gpu_ids)}) must match number of containers ({num_containers})"
        assert len(set(gpu_ids)) == len(gpu_ids), f"GPU IDs must be unique, got: {gpu_ids}"
    
    services = {}
    
    print(f"🔧 Generating Docker Compose file for {num_containers} containers...")
    print(f"   Ports: {start_port} to {start_port + num_containers - 1}")
    if mountable_volumes:
        print(f"   Mountable volumes: {mountable_volumes} -> /data (read-only)")
    
    if mountable_volume_saveto:
        print(f"   Save directories: {mountable_volume_saveto}/container_XXX -> /submission")
        # Create individual directories for each container
        os.makedirs(mountable_volume_saveto, exist_ok=True)
        for i in range(num_containers):
            container_save_dir = os.path.join(mountable_volume_saveto, f'container_{i:03d}')
            os.makedirs(container_save_dir, exist_ok=True)
        print(f"   ✅ Created {num_containers} container directories in {mountable_volume_saveto}")
    
    if extra_env:
        print(f"   Extra environment variables: {extra_env}")
    
    if gpu_ids:
        print(f"   GPU support: ENABLED (GPUs {gpu_ids} mapped to containers 0-{num_containers-1})")
    
    print(f"   Output: {output_file}")
    
    for i in range(num_containers):
        container_id = i
        host_port = start_port + i
        service_name = f'executor-{container_id:03d}'
        container_type = container_assignments[i]
        
        # Assign GPU ID if provided
        gpu_id = gpu_ids[i] if gpu_ids else None
        
        services[service_name] = generate_container_config(container_id, host_port, container_type, mountable_volumes, mountable_volume_saveto, extra_env, manager_url=None, gpu_id=gpu_id)
    
    # Add manager service
    services['manager'] = {
        'image': 'manager-prebuilt',
        'command': 'python main.py',
        'ports': ['5000:5000'],
        'environment': [
            'PORT=5000',
            'CONTAINER_CONFIG_PATH=container_config.json'
        ],
        'volumes': [
            f'./{config_file}:/app/container_config.json:ro'
        ],
        'depends_on': [f'executor-{i:03d}' for i in range(num_containers)],
        'restart': 'unless-stopped'
    }
    
    # Create the complete compose configuration
    compose_config = {
        'version': '3.8',
        'services': services,
        'networks': {
            'default': {
                'driver': 'bridge'
            }
        }
    }
    
    # Write the file
    with open(output_file, 'w') as f:
        yaml.dump(compose_config, f, default_flow_style=False, sort_keys=False, width=120)
    
    print(f"✅ Generated {output_file}")
    print(f"📊 Total containers: {num_containers}")
    print(f"📊 Port range: {start_port}-{start_port + num_containers - 1}")
    
    # Show usage instructions
    print("\n🚀 Usage:")
    print(f"   docker compose -f {output_file} up -d")
    print(f"   docker compose -f {output_file} ps")
    print(f"   docker compose -f {output_file} down")

# Default container configuration
DEFAULT_CONTAINER_CONFIG = {
    "image": "executor-prebuilt",
    "mem_limit": "256M",
    "cpus": 0.25
}

def parse_container_types(types_string: str, num_containers: int) -> dict:
    """Parse container types and return dict mapping container_id -> type_name"""
    
    # Parse "python:80,nodejs:20" format
    type_specs = {}
    for type_spec in types_string.split(','):
        if ':' not in type_spec:
            raise ValueError(f"Invalid type specification: {type_spec}. Use format 'type:count'")
        
        type_name, count_str = type_spec.split(':', 1)
        type_name = type_name.strip()
        count = int(count_str.strip())
        
        # Accept any container type name - expect it to be a tagged container
        type_specs[type_name] = count
    
    # Validate total
    total_specified = sum(type_specs.values())
    if total_specified != num_containers:
        raise ValueError(f"Container type counts ({total_specified}) don't match --num-containers ({num_containers}). "
                        f"Specified: {dict(type_specs)}")
    
    # Generate assignments
    assignments = {}
    container_id = 0
    for type_name, count in type_specs.items():
        for _ in range(count):
            assignments[container_id] = type_name
            container_id += 1
    
    return assignments

def parse_env_vars(env_string: str) -> dict:
    """Parse environment variables from string format KEY=VALUE,KEY2=VALUE2"""
    env_dict = {}
    if env_string:
        pairs = env_string.split(',')
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                env_dict[key.strip()] = value.strip()
    return env_dict

def parse_gpu_ids(gpu_string: str) -> list[int]:
    """Parse GPU IDs from comma-separated string"""
    gpu_ids = []
    if gpu_string:
        for gpu_id_str in gpu_string.split(','):
            gpu_ids.append(int(gpu_id_str.strip()))
    return gpu_ids

def main():
    parser = argparse.ArgumentParser(description='Generate Docker Compose file for multiple containers')
    parser.add_argument('--num-containers', '-n', type=int, default=10, help='Number of containers (default: 10)')
    parser.add_argument('--start-port', '-p', type=int, default=60000, help='Starting port number (default: 60000)')
    parser.add_argument('--output', '-o', default='docker-compose.yml', help='Output filename')
    parser.add_argument('--mountable-volumes', '-m', help='Mountable volumes directory')
    parser.add_argument('--mountable-volume-saveto', '-s', help='Mountable volume directory to save')
    parser.add_argument('--env', '-e', help='Extra environment variables in format KEY=VALUE,KEY2=VALUE2')
    parser.add_argument('--types', '-t', help='Container types in format python:10,nodejs:5 (exact counts)')
    parser.add_argument('--gpu-ids', '-g', help='GPU IDs to assign to containers, comma-separated (e.g., 0,1,2,3,4,5,6,7)')
    parser.add_argument('--config-file', '-c', default='container_config.json', help='Container config file name (default: container_config.json)')
    
    args = parser.parse_args()
    
    extra_env = parse_env_vars(args.env) if args.env else None
    gpu_ids = parse_gpu_ids(args.gpu_ids) if args.gpu_ids else None
    container_assignments = parse_container_types(args.types, args.num_containers)
    
    # Show type breakdown
    if args.types:
        type_counts = {}
        for container_type in container_assignments.values():
            type_counts[container_type] = type_counts.get(container_type, 0) + 1
        
        print("🔧 Container type breakdown:")
        for type_name, count in type_counts.items():
            percentage = (count / args.num_containers) * 100
            print(f"   {type_name}: {count} containers ({percentage:.1f}%)")
    
    generate_compose_file(
        num_containers=args.num_containers,
        container_assignments=container_assignments,
        start_port=args.start_port,
        output_file=args.output,
        mountable_volumes=args.mountable_volumes,
        mountable_volume_saveto=args.mountable_volume_saveto,
        extra_env=extra_env,
        gpu_ids=gpu_ids,
        config_file=args.config_file
    )
    
    generate_container_config_json(
        num_containers=args.num_containers,
        container_assignments=container_assignments,
        output_file=args.config_file
    )

if __name__ == "__main__":
    main() 