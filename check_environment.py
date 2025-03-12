#!/usr/bin/env python
"""
Script to check the environment and identify potential issues that could cause
inconsistencies between local development and deployment environments.
"""

import os
import sys
import platform
import logging
import json
from pathlib import Path
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check Python version and implementation."""
    version = sys.version
    implementation = platform.python_implementation()
    logger.info(f"Python version: {version}")
    logger.info(f"Python implementation: {implementation}")
    return {
        'version': version,
        'implementation': implementation
    }

def check_library_versions():
    """Check versions of key libraries."""
    libraries = [
        'numpy', 'pandas', 'sklearn', 'joblib', 'matplotlib', 
        'seaborn', 'django', 'whitenoise', 'gunicorn'
    ]
    
    versions = {}
    for lib in libraries:
        try:
            module = importlib.import_module(lib)
            version = getattr(module, '__version__', 'Unknown')
            logger.info(f"{lib} version: {version}")
            versions[lib] = version
        except ImportError:
            logger.warning(f"{lib} is not installed")
            versions[lib] = None
    
    return versions

def check_environment_variables():
    """Check relevant environment variables."""
    env_vars = [
        'DJANGO_SETTINGS_MODULE', 'DEBUG', 'SECRET_KEY', 
        'PYTHONPATH', 'RENDER', 'PORT'
    ]
    
    env_values = {}
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        # Mask sensitive values
        if var in ['SECRET_KEY']:
            if value != 'Not set':
                value = f"{value[:3]}...{value[-3:]} (masked for security)"
        logger.info(f"Environment variable {var}: {value}")
        env_values[var] = value
    
    return env_values

def check_file_paths():
    """Check important file paths."""
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = script_dir
    
    paths = {
        'script_dir': str(script_dir),
        'project_root': str(project_root),
        'current_working_dir': os.getcwd(),
        'python_path': sys.executable,
        'sys_path': sys.path
    }
    
    logger.info(f"Script directory: {paths['script_dir']}")
    logger.info(f"Project root: {paths['project_root']}")
    logger.info(f"Current working directory: {paths['current_working_dir']}")
    logger.info(f"Python executable: {paths['python_path']}")
    logger.info(f"sys.path: {paths['sys_path']}")
    
    # Check specific files
    files_to_check = [
        'diabetes_dataset.csv',
        'predictor/models/stacking_model.joblib',
        'predictor/models/scaler.joblib',
        'predictor/models/model_data.joblib',
        'predictor/ml_model.py',
        'manage.py',
        'requirements.txt'
    ]
    
    file_exists = {}
    for file in files_to_check:
        file_path = os.path.join(project_root, file)
        exists = os.path.exists(file_path)
        logger.info(f"File {file}: {'Exists' if exists else 'Does not exist'} at {file_path}")
        file_exists[file] = exists
    
    paths['file_exists'] = file_exists
    return paths

def check_platform_info():
    """Check platform information."""
    info = {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'node': platform.node()
    }
    
    logger.info(f"System: {info['system']}")
    logger.info(f"Release: {info['release']}")
    logger.info(f"Version: {info['version']}")
    logger.info(f"Machine: {info['machine']}")
    logger.info(f"Processor: {info['processor']}")
    logger.info(f"Node: {info['node']}")
    
    return info

def check_dataset():
    """Check the dataset file."""
    try:
        import pandas as pd
        
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = script_dir
        dataset_path = os.path.join(project_root, 'diabetes_dataset.csv')
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found at {dataset_path}")
            return {
                'exists': False,
                'path': dataset_path
            }
        
        data = pd.read_csv(dataset_path)
        shape = data.shape
        columns = data.columns.tolist()
        head_hash = hash(tuple(map(tuple, data.head(5).values.tolist())))
        
        logger.info(f"Dataset shape: {shape}")
        logger.info(f"Dataset columns: {columns}")
        logger.info(f"Dataset first 5 rows hash: {head_hash}")
        
        return {
            'exists': True,
            'path': dataset_path,
            'shape': shape,
            'columns': columns,
            'head_hash': head_hash
        }
    except Exception as e:
        logger.error(f"Error checking dataset: {str(e)}")
        return {
            'exists': False,
            'error': str(e)
        }

def main():
    """Run all environment checks."""
    try:
        logger.info("=== Environment Check ===")
        
        # Run all checks
        python_info = check_python_version()
        library_versions = check_library_versions()
        env_vars = check_environment_variables()
        file_paths = check_file_paths()
        platform_info = check_platform_info()
        dataset_info = check_dataset()
        
        # Compile all information
        env_info = {
            'python': python_info,
            'libraries': library_versions,
            'environment_variables': env_vars,
            'file_paths': file_paths,
            'platform': platform_info,
            'dataset': dataset_info,
            'timestamp': import_time_module().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to file
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        output_file = os.path.join(script_dir, 'environment_info.json')
        
        with open(output_file, 'w') as f:
            json.dump(env_info, f, indent=2, default=str)
        
        logger.info(f"Environment information saved to {output_file}")
        logger.info("Run this script in both environments (local and Render) and compare the output files.")
        
        return 0
    except Exception as e:
        logger.error(f"Error during environment check: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

def import_time_module():
    """Import time module to avoid circular imports."""
    import datetime
    return datetime.datetime.now()

if __name__ == "__main__":
    sys.exit(main()) 