### Initialize Modal

#1. Create an image with all the dependencies
#2. Create a volume to store the downloaded models and parameters
#3. Create a volume to store the outputs
#4. Create a function that will be used to run the code

import os, time, signal
import sys, random, string, re
import modal
from datetime import datetime

print("Importing modal...")

# Create a Modal app
app = modal.App("rfdiffusion")

print("Creating volumes...")

# Create volumes for storing models and outputs
models_volume = modal.Volume.from_name("rfdiffusion-models", create_if_missing=True)
outputs_volume = modal.Volume.from_name("rfdiffusion-outputs", create_if_missing=True)

print("Volumes created successfully!")

# Create image with all dependencies
image = (
    modal.Image.from_registry("python:3.11")
    .apt_install(["git", "aria2", "wget", "unzip"])
    # Add SE3Transformer, ananas, and colabdesign
    .run_commands(
        "git clone https://github.com/sokrypton/RFdiffusion.git /opt/RFdiffusion",
        "cd /opt/RFdiffusion/env/SE3Transformer && pip install .",
        "wget -qnc https://files.ipd.uw.edu/krypton/ananas -O /usr/local/bin/ananas",
        "chmod +x /usr/local/bin/ananas",
        # Add ColabDesign installation
        "git clone https://github.com/sokrypton/ColabDesign.git /opt/ColabDesign",
        "cd /opt/ColabDesign && pip install -e ."
    )
    .pip_install([
        # Install numpy<2 first to ensure all subsequent packages use this version
        "numpy<2.0.0",
        "jedi", "omegaconf", "hydra-core", "icecream", "pyrsistent", 
        "pynvml", "decorator", "torch==2.2.1", "psutil", "scipy",
        "matplotlib", "pandas", "requests", "tqdm", "seaborn", "biopython", "se3_transformer",
        "git+https://github.com/NVIDIA/dllogger#egg=dllogger",
        "git+https://github.com/sokrypton/ColabDesign.git@v1.1.1"
    ])
    # Install DGL with CUDA support but no dependencies to avoid NVIDIA package conflicts
    .run_commands(
        "pip install --no-dependencies dgl==2.0.0 -f https://data.dgl.ai/wheels/cu121/repo.html",
        "pip install --no-dependencies e3nn==0.3.3 opt_einsum_fx"
    )
    .env({"DGLBACKEND": "pytorch", "PYTHONPATH": "/opt/RFdiffusion"})
)

print("Image created successfully!")

@app.function(
    image=image,
    volumes={
        "/data/models": models_volume,
    },
    timeout=3600,  # 1 hour timeout for downloading everything
)
def initialize_volumes():
    """Initialize volumes by downloading all necessary models and code repositories"""
    import os
    import time
    import sys
    
    os.chdir("/data/models")
    
    def log_progress(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    log_progress("Starting initialization...")
    
    # RFdiffusion is already installed in the container at /opt/RFdiffusion
    log_progress("Using RFdiffusion from container image at /opt/RFdiffusion")
    
    if not os.path.isdir("colabdesign"):
        log_progress("Setting up ColabDesign...")
        os.system("ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign")
    else:
        log_progress("ColabDesign already set up. Skipping...")
    
    # Create params directory and download models
    if not os.path.isdir("params"):
        log_progress("Downloading parameters and models...")
        os.system("mkdir -p params")
        
        # Download files
        log_progress("Downloading schedules...")
        os.system("aria2c -q -x 16 https://files.ipd.uw.edu/krypton/schedules.zip")
        log_progress("Downloading Base_ckpt.pt...")
        os.system("aria2c -q -x 16 http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt")
        log_progress("Downloading Complex_base_ckpt.pt...")
        os.system("aria2c -q -x 16 http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt")
        log_progress("Downloading AlphaFold parameters...")
        os.system("aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar")
        
        # Extract AlphaFold parameters
        log_progress("Extracting AlphaFold parameters...")
        os.system("tar -xf alphafold_params_2022-12-06.tar -C params")
        
        # Create RFdiffusion models directory
        if not os.path.isdir("RFdiffusion/models"):
            log_progress("Creating RFdiffusion models directory...")
            os.system("mkdir -p RFdiffusion/models")
            
        # Move model files to RFdiffusion directory
        log_progress("Moving model files to RFdiffusion directory...")
        models = ["Base_ckpt.pt", "Complex_base_ckpt.pt"]
        os.system(f"mv {' '.join(models)} RFdiffusion/models")
        
        # Extract schedules
        log_progress("Extracting schedules...")
        os.system("unzip schedules.zip && rm schedules.zip")
        
        # Create a marker file to indicate completion
        with open("params/done.txt", "w") as f:
            f.write(f"Initialization completed at {datetime.now().isoformat()}")
        
        log_progress("All files downloaded and extracted successfully")
    else:
        log_progress("Params directory already exists. Skipping downloads...")
    
    log_progress("Initialization completed successfully!")
    return "Volumes initialized successfully"

# Example of a function that could use the initialized volumes and image
@app.function(
    image=image,
    volumes={
        "/data/models": models_volume,
        "/data/outputs": outputs_volume,
    },
    gpu="T4",
)
def run_rfdiffusion(input_data):
    """Run RFdiffusion with the input data"""
    import os
    import sys
    
    # Add RFdiffusion to path
    os.chdir("/data/models")
    sys.path.append('/data/models/RFdiffusion')
    
    # Your RFdiffusion code goes here
    return "RFdiffusion execution completed"

if __name__ == "__main__":
    # Run initialization when script is executed directly
    with app.run():
        initialize_volumes.remote()