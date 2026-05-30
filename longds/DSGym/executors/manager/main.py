import os
import json
import asyncio
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    outputs: list[Any]

class AllocationResponse(BaseModel):
    container_id: int

class ContainerConfig:
    def __init__(self):
        config_path = os.getenv("CONTAINER_CONFIG_PATH", "container_config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def get_url(self, container_id: int) -> str:
        container_key = str(container_id)
        if container_key not in self.config:
            raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
        
        return self.config[container_key]['url']
    
    def get_type(self, container_id: int) -> str:
        container_key = str(container_id)
        if container_key not in self.config:
            raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
        
        return self.config[container_key]['type']
    
    def get_containers_by_type(self, container_type: Optional[str] = None) -> list[int]:
        """Get list of container IDs of specific type"""
        containers = []
        for container_id_str in self.config.keys():
            container_id = int(container_id_str)
            if container_type is None or self.get_type(container_id) == container_type:
                containers.append(container_id)
        return containers
    
    @property
    def urls(self):
        """Backward compatibility property"""
        return self.config

config = ContainerConfig()

# Type-based container queues
available_containers_by_type = {}
allocated_containers = set()

# Initialize queues per container type
for container_id_str in config.config.keys():
    container_id = int(container_id_str)
    container_type = config.get_type(container_id)
    
    if container_type not in available_containers_by_type:
        available_containers_by_type[container_type] = asyncio.Queue()
    
    available_containers_by_type[container_type].put_nowait(container_id)

print("🔧 Initialized container queues:")
for container_type, queue in available_containers_by_type.items():
    print(f"   {container_type}: {queue.qsize()} containers")

http_limits = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
http_client = httpx.AsyncClient(timeout=18000.0, limits=http_limits)
ready_check_client = httpx.AsyncClient(timeout=15.0, limits=http_limits)

async def make_request(client: httpx.AsyncClient, method: str, container_id: int, endpoint: str, **kwargs):
    container_url = config.get_url(container_id)
    url = f"{container_url}/{endpoint}"
    
    try:
        response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail=f"Container {container_id} is not available")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Container error: {e.response.text}")

app = FastAPI(title="Executor Manager")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/allocate", response_model=AllocationResponse)
async def allocate_container(container_type: Optional[str] = None):
    # If no type specified, try to allocate from any available type
    if container_type is None:
        for type_name, queue in available_containers_by_type.items():
            if queue.qsize() > 0:
                try:
                    container_id = await asyncio.wait_for(queue.get(), timeout=0.1)
                    allocated_containers.add(container_id)
                    return AllocationResponse(container_id=container_id)
                except asyncio.TimeoutError:
                    continue
        raise HTTPException(status_code=503, detail="No containers available")
    
    # Allocate specific type
    if container_type not in available_containers_by_type:
        raise HTTPException(status_code=400, detail=f"Unknown container type: {container_type}")
    
    try:
        container_id = await asyncio.wait_for(available_containers_by_type[container_type].get(), timeout=30.0)
        allocated_containers.add(container_id)
        return AllocationResponse(container_id=container_id)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail=f"No {container_type} containers available")

@app.post("/deallocate/{container_id}")
async def deallocate_container(container_id: int):
    if container_id not in allocated_containers:
        raise HTTPException(status_code=400, detail=f"Container {container_id} is not allocated")
    
    await make_request(http_client, "POST", container_id, "restart")
    allocated_containers.discard(container_id)
    
    # Put container back in the appropriate type queue
    container_type = config.get_type(container_id)
    available_containers_by_type[container_type].put_nowait(container_id)
    
    return {"status": "deallocated", "container_id": container_id}

@app.post("/restart")
async def restart_all_allocated_containers():
    """Restart and deallocate all currently allocated containers"""
    if not allocated_containers:
        return {"status": "no_containers_to_restart", "restarted_containers": []}
    
    restarted_containers = []
    failed_containers = []
    
    # Create a copy of allocated_containers to iterate over
    containers_to_restart = list(allocated_containers)
    
    for container_id in containers_to_restart:
        try:
            # Restart the container
            await make_request(http_client, "POST", container_id, "restart")
            
            # Remove from allocated set
            allocated_containers.discard(container_id)
            
            # Put container back in the appropriate type queue
            container_type = config.get_type(container_id)
            available_containers_by_type[container_type].put_nowait(container_id)
            
            restarted_containers.append(container_id)
        except Exception as e:
            # If restart fails, still remove from allocated set to avoid stuck containers
            allocated_containers.discard(container_id)
            failed_containers.append({"container_id": container_id, "error": str(e)})
    
    return {
        "status": "restart_completed",
        "restarted_containers": restarted_containers,
        "failed_containers": failed_containers,
        "total_restarted": len(restarted_containers),
        "total_failed": len(failed_containers)
    }

@app.post("/session/{container_id}/execute")
async def execute_code(container_id: int, request: CodeRequest):
    if container_id not in allocated_containers:
        raise HTTPException(status_code=400, detail=f"Container {container_id} is not allocated or invalid")
    return await make_request(http_client, "POST", container_id, "execute", json={"code": request.code})

@app.get("/session/{container_id}/health")
async def container_health(container_id: int):
    return await make_request(ready_check_client, "GET", container_id, "health")

@app.get("/session/{container_id}/ready")
async def container_ready(container_id: int):
    return await make_request(ready_check_client, "GET", container_id, "ready")

@app.post("/session/{container_id}/restart")
async def restart_container(container_id: int):
    return await make_request(http_client, "POST", container_id, "restart")

@app.get("/session/{container_id}/restart-status")
async def restart_status(container_id: int):
    return await make_request(ready_check_client, "GET", container_id, "restart-status")

@app.api_route("/session/{container_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_container(container_id: int, path: str, request: Request):
    kwargs = {}
    
    if request.query_params:
        kwargs['params'] = request.query_params
    
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        if body:
            kwargs['content'] = body
    
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length"]}
    if headers:
        kwargs['headers'] = headers
    
    return await make_request(http_client, request.method, container_id, path, **kwargs)

@app.get("/status")
async def get_status():
    available_by_type = {}
    total_available = 0
    for container_type, queue in available_containers_by_type.items():
        count = queue.qsize()
        available_by_type[container_type] = count
        total_available += count
    
    return {
        "available_containers": total_available,
        "available_by_type": available_by_type,
        "allocated_containers": len(allocated_containers),
        "total_containers": len(config.config),
        "allocated_container_ids": list(allocated_containers)
    }

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
    await ready_check_client.aclose()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port) 