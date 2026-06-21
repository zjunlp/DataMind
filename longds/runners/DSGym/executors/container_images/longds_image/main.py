import os
from fastapi import FastAPI
from pydantic import BaseModel
from kernel_executor import KernelExecutor

app = FastAPI()

class CodeRequest(BaseModel):
    code: str

timeout_seconds = int(os.getenv("EXECUTION_TIMEOUT", "180"))
executor = KernelExecutor(timeout=timeout_seconds)

@app.on_event("startup")
async def startup_event():
    executor.start()

@app.on_event("shutdown") 
async def shutdown_event():
    executor.stop()

@app.post("/execute")
async def execute_code(request: CodeRequest):
    outputs = executor.execute(request.code)
    return {"outputs": outputs}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    """Check if the Jupyter kernel is ready for code execution"""
    if executor.km is None or executor.kc is None:
        return {"ready": False, "reason": "kernel_not_started"}
    
    if executor.is_restarting:
        return {"ready": False, "reason": "kernel_restarting"}
    
    try:
        # Execute a simple test and verify we get the correct result
        outputs = executor.execute("1+1")
        
        # Check if we got the expected result
        if outputs and len(outputs) > 0:
            # Look for the result output with value 2
            has_correct_result = any(
                output.get("type") == "result" and 
                output.get("data", {}).get("text/plain") == "2"
                for output in outputs
            )
            if has_correct_result:
                return {"ready": True}
            else:
                return {"ready": False, "reason": f"Unexpected result: {outputs}"}
        else:
            return {"ready": False, "reason": "No output from test execution"}
    except Exception as e:
        return {"ready": False, "reason": str(e)}

@app.post("/restart")
async def restart_kernel():
    """Non-blocking restart of the Jupyter kernel"""
    try:
        result = executor.restart()
        return result
    except Exception as e:
        return {"status": "error", "reason": str(e)}

@app.get("/restart-status")
async def restart_status():
    """Check if kernel is currently restarting"""
    return {"restarting": executor.is_restarting}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8432))
    uvicorn.run(app, host="0.0.0.0", port=port) 