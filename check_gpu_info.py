import subprocess

def get_gpu_info():
    try:
        # Get GPU information using nvidia-smi command
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.stdout
    except FileNotFoundError:
        return "NVIDIA GPU not found or NVIDIA drivers not installed"
    except Exception as e:
        return f"Error checking GPU: {str(e)}"

print("GPU Information:")
print(get_gpu_info())
