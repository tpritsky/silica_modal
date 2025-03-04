#!/usr/bin/env python3
"""
Interactive version of basic_test.py for debugging and package installation
"""

import modal

# Import your existing app and resources
from initialize_modal import app, models_volume, outputs_volume, image

# Import the function definition without executing it
from basic_test import run_rfdiffusion_test as original_test_func

# Create an interactive version with the same signature
@app.function(
    image=image,
    volumes={
        "/data/models": models_volume,
        "/data/outputs": outputs_volume,
    },
    gpu="T4",
    # The 'interactive' parameter is not supported in your Modal version
    # We'll remove it and use regular 'modal shell' functionality
)
def interactive_test(
    name="test",
    contigs="100",
    pdb_content=None,
    iterations=50,
    symmetry="none",
    order=1,
    hotspot=None,
    chains=None,
    add_potential=True,
    num_designs=1,
):
    """Interactive version of run_rfdiffusion_test for debugging"""
    import os
    import sys
    
    # Setup initial environment
    os.chdir("/data/models")
    sys.path.append('/data/models/RFdiffusion')
    
    # Create helpful debug scripts
    with open("/tmp/test_imports.py", "w") as f:
        f.write("""
import os
import sys
import time

# Add RFdiffusion to path
os.chdir("/data/models")
sys.path.append('/data/models/RFdiffusion')

# Try importing necessary modules
modules = [
    "inference.utils", 
    "colabdesign.rf.utils",
    "colabdesign.shared.protein"
]

for module in modules:
    try:
        exec(f"import {module}")
        print(f"✅ Successfully imported {module}")
    except ImportError as e:
        print(f"❌ Failed to import {module}: {e}")
        
print("\\nIf imports are failing, try installing the required packages with:")
print("pip install <package_name>")
""")

    with open("/tmp/run_test.py", "w") as f:
        f.write("""
# This script runs the actual test function with the current parameters
# You can modify this file to change test parameters

import os
import sys
import random
import string
import time

# Add RFdiffusion to path
os.chdir("/data/models")
sys.path.append('/data/models/RFdiffusion')

# Try running a basic RFdiffusion command
name = "test"
contigs = "100"
iterations = 50
symmetry = "none"
order = 1
hotspot = None
chains = None
add_potential = True
num_designs = 1

# Generate output path
path = name + "_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
full_path = f"/data/outputs/{path}"
os.makedirs(full_path, exist_ok=True)

# Build options
opts = [
    f"inference.output_prefix={full_path}",
    f"inference.num_designs={num_designs}",
    f"diffuser.T={iterations}",
    f"'contigmap.contigs=[{contigs}]'"
]

# Run command
cmd = f"cd /data/models && python RFdiffusion/run_inference.py {' '.join(opts)}"
print(f"Running command: {cmd}")
    
start_time = time.time()
result = os.system(cmd)
end_time = time.time()

print(f"Result: {'Success' if result == 0 else 'Failed'} (code {result})")
print(f"Runtime: {end_time - start_time:.2f} seconds")
""")

    print("\n" + "="*80)
    print("INTERACTIVE DEBUGGING SESSION")
    print("="*80)
    print("\nYou can now debug and install packages in this environment.")
    print("\nUseful commands:")
    print("  python /tmp/test_imports.py  - Test required module imports")
    print("  python /tmp/run_test.py      - Run a test RFdiffusion job")
    print("  pip install <package>        - Install missing packages")
    print("  pip freeze > /tmp/packages.txt; cat /tmp/packages.txt  - List installed packages")
    print("\nWhen finished, type 'exit' to leave the interactive session.")
    print("="*80)
    
    # In non-interactive mode, this function will actually run to completion
    return "Interactive session script completed"

if __name__ == "__main__":
    print("Starting interactive debugging session...")
    print("Use 'modal shell interactive_test.interactive_test' to start an interactive shell")
    
    with app.run():
        # This will just run the function normally if executed directly
        result = interactive_test.remote()
        print(result) 