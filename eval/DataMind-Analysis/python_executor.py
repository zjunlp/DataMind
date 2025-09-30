import code
import collections
import contextlib
import copy
import io
import json
import logging
import os
import signal
import time
import uuid
import re
import sys
import subprocess
from types import ModuleType
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats
from scipy.stats import pearsonr, ttest_ind, norm
from sklearn.preprocessing import LabelEncoder
from statsmodels.formula.api import logit, ols
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
import seaborn as sns
from statsmodels.regression.linear_model import OLS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import linregress
# from econml.dml import DML
from statsmodels.stats.proportion import proportion_confint
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
from scipy.stats import bootstrap
from scipy.stats import chisquare
from sklearn.neighbors import NearestNeighbors
# from collections.abc import Mapping
# from collections import namedtuple
from pingouin import partial_corr
import pingouin as pg
from tabulate import tabulate

logger = logging.getLogger(__name__)

IMPORT_HELPER = {
    "python": [
        # Basic Python library
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        # "import itertools",
        # "import collections",
        # "import heapq",
        "import statistics",
        "import functools",
        # "import hashlib",
        # "import pingouin",
        
        # Data processing and scientific computing
        "import numpy",
        "import numpy as np",
        "import pandas",  
        "import pandas as pd",
        "import scipy.stats as stats",
        "from scipy import stats",
        "from scipy.stats import pearsonr",
        "from scipy.stats import norm",
        "from scipy.stats import chi2_contingency",
        
        # Statistical modeling
        "import statsmodels.api as sm",
        "from statsmodels.formula.api import ols",
        "from statsmodels.tsa.stattools import adfuller",
        "from statsmodels.tsa.stattools import grangercausalitytests",
        "from statsmodels.stats.outliers_influence import variance_inflation_factor",
        
        # Machine learning
        "from sklearn.linear_model import LinearRegression",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.preprocessing import StandardScaler",

        # Visualization
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import networkx as nx",
        
        # Other commonly used tools
        "import openpyxl",  # Excel file support
        "import string",
    ],
}

class TimeOutException(Exception):
    pass

def handler(signum, frame):
    logger.info("Code execution timeout")
    raise TimeOutException

class CodeRunner:
    def __init__(self, timeout=120):
        self.interpreter = code.InteractiveInterpreter()
        self.timeout = timeout
        self.locals = None
        # self.base_path = base_path

        # Initialize interpreter environment
        self._initialize_interpreter()
        
    def _initialize_interpreter(self):
        """Initialize interpreter environment and import necessary packages"""
        # Execute all predefined import statements
        for import_stmt in IMPORT_HELPER["python"]:
            try:
                self.interpreter.runsource(import_stmt)
            except Exception as e:
                logger.warning(f"Failed to execute import: {import_stmt}")
                logger.warning(f"Error: {str(e)}")


    def run_code(self, python_code, base_path= None):
        """
        Execute the Python code and return the result
        
        Args:
            python_code: The Python code to execute

        Returns:
            tuple: (output string, error message, whether has error)
        """
        try:
            # Handle file paths
            if base_path is not None:
                import os
                os.chdir(base_path)
            else:
                exit()

            # Set timeout handling
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.timeout)


            # Execute code and capture output
            with contextlib.redirect_stdout(io.StringIO()) as fout, contextlib.redirect_stderr(io.StringIO()) as ferr:
                self.interpreter.runcode(python_code)
            signal.alarm(0)

            # Process output and error
            out = fout.getvalue().strip()
            err = ferr.getvalue().strip()
            err = err.replace(os.environ.get("HOME", ""), "~")
            err = err.replace(os.getcwd(), ".")

            if "The above exception was the direct cause of the following exception:" in err:
                error_parts = err.split("The above exception was the direct cause of the following exception:")
                err = error_parts[-1].strip()

            return out, err, err != ""
            
        except TimeOutException:
            return "", "Code execution timed out", True
        except Exception as e:
            return "", str(e), True

    def get_state(self):
        """
        Get interpreter state and return a snapshot of the current state
        
        Returns:
            dict: A dictionary containing the current state of the interpreter, or None if failed
        """
        try:
            state = {}
            for k, v in self.interpreter.locals.items():
                if k == '__builtins__':
                    continue
                try:
                    if isinstance(v, ModuleType):
                        state[k] = v
                    elif isinstance(v, collections.abc.KeysView):
                        state[k] = list(v)
                    elif isinstance(v, pd.io.excel._base.ExcelFile):
                        state[k] = {'_excel_file_path': v.io}
                    elif hasattr(v, 'to_dict'):  # Handle pandas objects
                        state[k] = v.to_dict()
                    else:
                        state[k] = copy.deepcopy(v)
                except Exception as e:
                    logger.warning(f"Error backing up interpreter when copying {k}: {e}")
                    logger.warning(f"v type is {type(v)}")
                    state[k] = f"UNCOPYABLE_OBJECT_{type(v).__name__}"
            return state
        except Exception as e:
            logger.warning(f"Error in get_state: {e}")
            return None

    def set_state(self, state, code2execute=None):
        """
        Set interpreter state, if state is invalid, re-execute code2execute

        Args:
            state: The dictionary of the state to be restored
            code2execute: A list containing historical execution code, each element should include a 'code' key

        Returns:
            bool: Whether the state restoration was successful
        """
        try:
            if state is not None:
                self.interpreter.locals.clear()
                for k, v in state.items():
                    if isinstance(v, str) and v.startswith("UNCOPYABLE_OBJECT_"):
                        logger.warning(f"Uncopyable object detected for {k}. This variable will be undefined.")
                        continue
                    try:
                        if isinstance(v, dict):
                            if '_excel_file_path' in v:
                                self.interpreter.locals[k] = pd.ExcelFile(v['_excel_file_path'])
                            else:
                                try:
                                    self.interpreter.locals[k] = pd.DataFrame.from_dict(v)
                                except:
                                    self.interpreter.locals[k] = v
                        else:
                            self.interpreter.locals[k] = v
                    except Exception as e:
                        logger.warning(f"Error restoring state for {k}: {e}")
                return True
            
            # If the status is invalid and code2execute is provided, re-execute the historical code
            elif code2execute:
                logger.warning("Invalid state, re-executing code from code2execute...")
                self.interpreter.locals.clear()
                for entry in code2execute:
                    if 'code' in entry:
                        out, err, has_error = self.run_code(entry['code'])
                        if has_error:
                            logger.error(f"Error re-executing code: {err}")
                            return False
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error in set_state: {e}")
            return False

    def clear_state(self):
        """Clear interpreter state"""
        self.interpreter.locals.clear()
        self._initialize_interpreter()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test code
    runner = CodeRunner()
    
    # Test the execution of basic code
    test_code = """
import pandas as pd
import numpy as np

# Create test data
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c']
})
x = 42
arr = np.array([1, 2, 3])
# All these paths will be replaced with <base_path + filename>
import pandas as pd
df = pd.read_csv("minwage.csv")
df1 = pd.read_csv('./data/ak91.csv')
df2 = pd.read_csv('../datathis/avandia.csv')
    """
    
    print("=== Testing basic code execution ===")
    out, err, has_error = runner.run_code(test_code)
    print("Output:", out)
    print("Error:", err)
    print("Has error:", has_error)

    # Test state management
    print("\n=== Testing state management ===")
    print("1. Getting current state...")
    state = runner.get_state()
    print("State keys:", state.keys() if state else "No state")
    
    print("\n2. Clearing state...")
    runner.clear_state()
    
    print("\n3. Verifying state is cleared...")
    out, err, has_error = runner.run_code("print(x if 'x' in locals() else 'x not found')")
    print("Output:", out)
    
    print("\n4. Restoring state...")
    success = runner.set_state(state)
    print("State restoration successful:", success)
    
    print("\n5. Verifying state is restored...")
    verify_code = """
print('Variable x:', x)
print('DataFrame df:')
print(df)
print('NumPy array arr:')
print(arr)
    """
    out, err, has_error = runner.run_code(verify_code)
    print("Output:", out)
    print("Error:", err)
    print("Has error:", has_error)

    # Test code history recovery
    print("\n=== Testing code history recovery ===")
    runner.clear_state()
    code_history = [{"code": test_code}]
    success = runner.set_state(None, code_history)
    print("Code history restoration successful:", success)
    
    if success:
        out, err, has_error = runner.run_code(verify_code)
        print("Output:", out)
        print("Error:", err)
        print("Has error:", has_error)
