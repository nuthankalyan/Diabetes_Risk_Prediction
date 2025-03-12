#!/usr/bin/env python
"""
Script to check deployment readiness and troubleshoot common issues.
Run this script before deploying to Render to ensure all requirements are met.
"""

import os
import sys
import logging
import importlib.util
import subprocess
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_file_exists(filepath, required=True):
    """Check if a file exists and log the result."""
    exists = os.path.isfile(filepath)
    status = "✅" if exists else "❌" if required else "⚠️"
    logger.info(f"{status} {filepath} {'exists' if exists else 'does not exist'}")
    return exists

def check_directory_exists(dirpath, required=True):
    """Check if a directory exists and log the result."""
    exists = os.path.isdir(dirpath)
    status = "✅" if exists else "❌" if required else "⚠️"
    logger.info(f"{status} {dirpath} {'exists' if exists else 'does not exist'}")
    return exists

def check_package_installed(package_name):
    """Check if a Python package is installed."""
    spec = importlib.util.find_spec(package_name)
    installed = spec is not None
    status = "✅" if installed else "❌"
    logger.info(f"{status} Package {package_name} {'is installed' if installed else 'is not installed'}")
    return installed

def check_settings_configuration():
    """Check Django settings configuration for deployment readiness."""
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from diabetes_prediction import settings
        
        # Check ALLOWED_HOSTS
        render_hosts = any(host in ['.render.com', '*.onrender.com'] for host in settings.ALLOWED_HOSTS)
        status = "✅" if render_hosts else "❌"
        logger.info(f"{status} ALLOWED_HOSTS {'includes' if render_hosts else 'does not include'} Render domains")
        
        # Check CSRF settings
        csrf_origins = hasattr(settings, 'CSRF_TRUSTED_ORIGINS') and any('render.com' in origin for origin in settings.CSRF_TRUSTED_ORIGINS)
        status = "✅" if csrf_origins else "❌"
        logger.info(f"{status} CSRF_TRUSTED_ORIGINS {'includes' if csrf_origins else 'does not include'} Render domains")
        
        # Check static files configuration
        static_root = hasattr(settings, 'STATIC_ROOT') and settings.STATIC_ROOT
        status = "✅" if static_root else "❌"
        logger.info(f"{status} STATIC_ROOT {'is configured' if static_root else 'is not configured'}")
        
        # Check whitenoise configuration
        whitenoise_middleware = 'whitenoise.middleware.WhiteNoiseMiddleware' in settings.MIDDLEWARE
        status = "✅" if whitenoise_middleware else "❌"
        logger.info(f"{status} WhiteNoise middleware {'is configured' if whitenoise_middleware else 'is not configured'}")
        
        # Check DEBUG setting
        debug_env = os.environ.get('DEBUG', 'True')
        status = "✅" if debug_env == 'False' else "⚠️"
        logger.info(f"{status} DEBUG is set to {debug_env} (should be 'False' for production)")
        
        return render_hosts and csrf_origins and static_root and whitenoise_middleware
    except ImportError as e:
        logger.error(f"Error importing settings: {str(e)}")
        return False

def check_build_script():
    """Check if build.sh exists and has the correct permissions."""
    build_script = 'build.sh'
    exists = check_file_exists(build_script)
    
    if exists:
        # Check if the script is executable (on Unix-like systems)
        if sys.platform != 'win32':
            try:
                executable = os.access(build_script, os.X_OK)
                status = "✅" if executable else "❌"
                logger.info(f"{status} {build_script} {'is executable' if executable else 'is not executable'}")
                
                if not executable:
                    logger.info("To make the script executable, run: chmod +x build.sh")
            except Exception as e:
                logger.error(f"Error checking script permissions: {str(e)}")
        else:
            logger.info("⚠️ Running on Windows - cannot check if build.sh is executable")
            logger.info("Note: On Render (Linux), the script needs to be executable")
    
    return exists

def check_procfile():
    """Check if Procfile exists and has the correct configuration."""
    procfile = 'Procfile'
    exists = check_file_exists(procfile)
    
    if exists:
        try:
            with open(procfile, 'r') as f:
                content = f.read().strip()
                correct_format = content.startswith('web: gunicorn')
                status = "✅" if correct_format else "❌"
                logger.info(f"{status} Procfile {'has correct format' if correct_format else 'has incorrect format'}")
                return correct_format
        except Exception as e:
            logger.error(f"Error reading Procfile: {str(e)}")
            return False
    
    return False

def check_render_yaml():
    """Check if render.yaml exists and has valid configuration."""
    render_yaml = 'render.yaml'
    exists = check_file_exists(render_yaml, required=False)
    
    if exists:
        try:
            with open(render_yaml, 'r') as f:
                try:
                    config = yaml.safe_load(f)
                    valid = isinstance(config, dict) and 'services' in config
                    status = "✅" if valid else "❌"
                    logger.info(f"{status} render.yaml {'has valid configuration' if valid else 'has invalid configuration'}")
                    return valid
                except Exception as e:
                    logger.error(f"Error parsing render.yaml: {str(e)}")
                    return False
        except Exception as e:
            logger.error(f"Error reading render.yaml: {str(e)}")
            return False
    
    return True  # Not required, so return True if not present

def check_requirements():
    """Check if requirements.txt exists and contains necessary packages."""
    requirements_file = 'requirements.txt'
    exists = check_file_exists(requirements_file)
    
    if exists:
        try:
            with open(requirements_file, 'r') as f:
                content = f.read()
                required_packages = ['django', 'gunicorn', 'whitenoise']
                missing_packages = [pkg for pkg in required_packages if pkg.lower() not in content.lower()]
                
                if missing_packages:
                    logger.warning(f"❌ requirements.txt is missing these packages: {', '.join(missing_packages)}")
                    return False
                else:
                    logger.info("✅ requirements.txt contains all necessary packages")
                    return True
        except Exception as e:
            logger.error(f"Error reading requirements.txt: {str(e)}")
            return False
    
    return False

def check_dataset():
    """Check if the dataset file exists."""
    dataset_file = 'diabetes_dataset.csv'
    return check_file_exists(dataset_file)

def check_model_files():
    """Check if pre-trained model files exist."""
    models_dir = 'predictor/models'
    dir_exists = check_directory_exists(models_dir, required=False)
    
    if dir_exists:
        model_file = os.path.join(models_dir, 'stacking_model.joblib')
        scaler_file = os.path.join(models_dir, 'scaler.joblib')
        model_data_file = os.path.join(models_dir, 'model_data.joblib')
        
        model_exists = check_file_exists(model_file, required=False)
        scaler_exists = check_file_exists(scaler_file, required=False)
        model_data_exists = check_file_exists(model_data_file, required=False)
        
        if model_exists and scaler_exists and model_data_exists:
            logger.info("✅ Pre-trained model files exist")
            return True
        else:
            logger.info("⚠️ Some model files are missing. They will be generated during deployment.")
            return False
    else:
        logger.info("⚠️ Models directory does not exist. It will be created during deployment.")
        return False

def main():
    """Run all deployment checks."""
    logger.info("=== Diabetes Prediction App Deployment Checker ===")
    
    # Check required files
    files_ok = (
        check_requirements() and
        check_build_script() and
        check_procfile() and
        check_render_yaml() and
        check_dataset()
    )
    
    # Check Django settings
    settings_ok = check_settings_configuration()
    
    # Check model files
    model_ok = check_model_files()
    
    # Check required packages
    packages_ok = all([
        check_package_installed('django'),
        check_package_installed('gunicorn'),
        check_package_installed('whitenoise'),
        check_package_installed('pandas'),
        check_package_installed('sklearn'),
        check_package_installed('joblib')
    ])
    
    # Summary
    logger.info("\n=== Deployment Readiness Summary ===")
    
    if files_ok and settings_ok and packages_ok:
        logger.info("✅ Your application is ready for deployment!")
    else:
        logger.warning("⚠️ Your application has some issues that need to be fixed before deployment.")
    
    if not model_ok:
        logger.info("\nℹ️ Pre-trained model files not found. You have two options:")
        logger.info("1. Let the model train during deployment (may take longer)")
        logger.info("2. Pre-train the model locally using: python pretrain_model.py")
    
    return 0

if __name__ == "__main__":
    try:
        # Try to import yaml for render.yaml validation
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed. Will not validate render.yaml structure.")
        yaml = None
        
    sys.exit(main()) 