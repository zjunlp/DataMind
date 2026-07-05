import os
import threading
import time
from jupyter_client import KernelManager
from output_cleaning import clean_jupyter_output

class KernelExecutor:
    def __init__(self, timeout: int = 18000):
        self.km = None
        self.kc = None
        self.code_folder = "/code"
        self._lock = threading.Lock()
        self._restarting = False
        self._restart_thread = None
        self.timeout = timeout
    
    def start(self):
        """
        Start the kernel. Uses locking to prevent conflicts with other threads
        """
        with self._lock:
            self.km = KernelManager()
            self.km.start_kernel()
            self.kc = self.km.blocking_client()
            
            if self.code_folder and os.path.exists(self.code_folder):
                self._setup_code_folder()
    
    def _setup_code_folder(self):
        """This is an helper function to load custom code into the kernel."""
        if not os.path.exists(self.code_folder):
            return
        
        if not self.kc:
            return
        
        # Wait for kernel to be ready before loading files
        try:
            self.kc.wait_for_ready(timeout=10)
        except Exception as e:
            print(f"Kernel not ready for code loading: {e}")
            return
        
        loaded_files = []
        
        for item in os.listdir(self.code_folder):
            if item.endswith('.py') and not item.startswith('_'):
                file_path = os.path.join(self.code_folder, item)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    
                    msg_id = self.kc.execute(file_content, silent=False)
                    
                    self.kc.wait_for_ready(timeout=5)
                    
                    loaded_files.append(item)
                    print(f"Loaded {item} successfully")
                    
                except Exception as e:
                    print(f"Could not load {item}: {e}")
        
        print(f"Code folder setup completed. Loaded files: {loaded_files}")
    
    def stop(self):
        with self._lock:
            if self.kc:
                self.kc.stop_channels()
            if self.km:
                self.km.shutdown_kernel()
    
    def restart(self):
        """Non-blocking restart of the kernel"""
        with self._lock:
            if self._restarting:
                return {"status": "already_restarting"}
            
            self._restarting = True
            self._restart_thread = threading.Thread(target=self._restart_worker)
            self._restart_thread.daemon = True
            self._restart_thread.start()
            
            return {"status": "restarting"}
    
    def _restart_worker(self):
        """Worker thread for restarting the kernel"""
        try:
            print("Restarting kernel...")
            self.stop()
            time.sleep(0.5)  # Brief pause to ensure cleanup
            self.start()
            print("Kernel restart completed")
        except Exception as e:
            print(f"Error during restart: {e}")
        finally:
            with self._lock:
                self._restarting = False
    
    @property
    def is_restarting(self):
        """Check if kernel is currently restarting"""
        with self._lock:
            return self._restarting
    
    def execute(self, code: str):
        """
        Execute code in the kernel.
        Args:
            code: The code to execute.
        Returns:
            A list of outputs.
        """
        verbose = os.getenv("VERBOSE_JUPYTER_LOG", "0") == "1"
        if verbose:
            print(f"DEBUG: Executing code with timeout={self.timeout}s: {code[:50]}...")
        
        while self.is_restarting:
            time.sleep(0.1)
        
        # Check if kernel is available
        if not self.kc:
            return [{"type": "error", "name": "KernelError", "value": "Kernel not available", "traceback": []}]
        
        if not self.km:
            return [{"type": "error", "name": "KernelError", "value": "Kernel not available", "traceback": []}]
        
        try:
            self.kc.wait_for_ready(timeout=10)
        except Exception as e:
            print(f"Kernel not ready: {e}")
            return [{"type": "error", "name": "KernelError", "value": "Kernel not ready", "traceback": []}]
        
        with self._lock:
            msg_id = self.kc.execute(code, silent=False)
            outputs = []
            
            # Track timeout state using mutable object
            timeout_state = {"occurred": False}
            
            # Set up timeout handler
            def timeout_handler():
                timeout_state["occurred"] = True
                try:
                    self.km.interrupt_kernel()
                    print(f"DEBUG: Execution timed out after {self.timeout} seconds, scheduling kernel restart")
                    # Schedule a kernel restart after timeout to ensure clean state
                    def delayed_restart():
                        time.sleep(1)  # Give interrupt time to take effect
                        print("DEBUG: Restarting kernel after timeout")
                        self.restart()
                    restart_thread = threading.Thread(target=delayed_restart)
                    restart_thread.daemon = True
                    restart_thread.start()
                except Exception as e:
                    print(f"DEBUG: Failed to interrupt kernel: {e}")

            # Set up timer for timeout
            timer = threading.Timer(self.timeout, timeout_handler)
            timer.start()
            
            start_time = time.time()

            while time.time() - start_time < self.timeout and not timeout_state["occurred"]:
                try:
                    msg = self.kc.get_iopub_msg(timeout=1)
                    
                    if msg['parent_header'].get('msg_id') == msg_id:
                        msg_type = msg['msg_type']
                        content = msg['content']
                        if verbose:
                            print(f"DEBUG: Received message type: {msg_type}")
                        
                        if msg_type == 'execute_result':
                            outputs.append({
                                'type': 'result',
                                'data': content['data']
                            })
                        elif msg_type == 'stream':
                            outputs.append({
                                'type': 'stream',
                                'name': content['name'],
                                'text': content['text']
                            })
                        elif msg_type == 'error':
                            outputs.append({
                                'type': 'error',
                                'name': content['ename'],
                                'value': content['evalue'],
                                'traceback': content['traceback']
                            })
                        elif msg_type == 'status' and content['execution_state'] == 'idle':
                            print("DEBUG: Execution completed normally")
                            break
                            
                except Exception as e:
                    # Ignore frequent poll timeouts to avoid log spam; emit only when verbose
                    if verbose:
                        print(f"DEBUG: Exception in message handling: {e}, msg")
                    continue  # Continue the loop for most exceptions
            
            timer.cancel()
            
            # Check if we hit timeout
            elapsed_time = time.time() - start_time
            timed_out = elapsed_time >= self.timeout or timeout_state["occurred"]
            
            if verbose:
                print(f"DEBUG: Elapsed time: {elapsed_time:.2f}s, Timeout occurred: {timeout_state['occurred']}, Timed out: {timed_out}")
                print(f"DEBUG: Number of outputs collected: {len(outputs)}")
            
            # If we hit timeout, add timeout error
            if timed_out:
                outputs.append({
                    'type': 'error',
                    'name': 'TimeoutError',
                    'value': f'Execution timed out after {self.timeout} seconds',
                    'traceback': []
                })
        
        return clean_jupyter_output(outputs) 
