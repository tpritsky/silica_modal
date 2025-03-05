#!/usr/bin/env python3
"""
Basic test for running RFdiffusion via Modal
"""

import os
import sys
import time
import random
import string
from datetime import datetime
from pathlib import Path
from typing import List

import modal

# Import from your Modal initialization file
from initialize_modal import app, models_volume, outputs_volume, image

# This function runs locally to read the PDB file and pass its contents to Modal
def run_rfdiffusion_with_local_pdb(
    name="test",
    batch_name=None,  # Added batch_name parameter
    contigs_list=None,
    pdb_path=None,
    iterations=50,
    symmetry="none",
    order=1,
    hotspot=None,
    chains=None,
    add_potential=True,
    num_designs=1,
):
    """Run RFdiffusion with a local PDB file"""
    # Generate batch name if not provided
    if batch_name is None:
        batch_name = f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Use default contig if none provided
    if contigs_list is None:
        contigs_list = ["100"]
    
    # Handle local PDB file if provided
    pdb_content = None
    if pdb_path and os.path.exists(pdb_path):
        print(f"Reading local PDB file: {pdb_path}")
        with open(pdb_path, "r") as f:
            pdb_content = f.read()
    
    # Create inputs for parallel execution over contigs AND designs
    inputs = []
    print(f"Running {len(contigs_list)} contigs with {num_designs} designs each...")
    for contigs in contigs_list:
        for design_num in range(num_designs):
            inputs.append((
                name,
                batch_name,  # Pass batch_name
                contigs,
                pdb_content,
                iterations,
                symmetry,
                order,
                hotspot,
                chains,
                add_potential,
                1,
                design_num,
            ))
    
    # Run in parallel using starmap and collect all results
    print(f"Running {len(inputs)} total designs in parallel...")
    results = list(run_rfdiffusion_test.starmap(inputs))
    print(f"All runs completed in batch: {batch_name}")
    print(f"Generated {len(results)} output folders:")
    for result in results:
        print(f"  {result['folder_name']}")
        print(f"  MPNN args: {result['mpnn_args']}")
    return batch_name, results  # Return both batch name and results

@app.function(
    image=image,
    volumes={
        "/data/models": models_volume,
        "/data/outputs": outputs_volume,
    },
    gpu="A100",
    timeout=14400,
)
def run_rfdiffusion_test(
    name="test",
    batch_name="default_batch",
    contigs="100",
    pdb_content=None,
    iterations=50,
    symmetry="none",
    order=1,
    hotspot=None,
    chains=None,
    add_potential=True,
    num_designs=1,
    design_num=0,
):
    """Run RFdiffusion with the specified parameters"""
    import os
    import sys
    import time
    from pathlib import Path
    
    # Create batch directory
    batch_path = f"/data/outputs/{batch_name}"
    os.makedirs(batch_path, exist_ok=True)
    
    # Generate unique folder name within batch
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    folder_name = f"{name}_contig{contigs.replace('/', '-')}_design{design_num}_{timestamp}_{run_id}"
    run_path = f"{batch_path}/{folder_name}"
    
    # Create run directory and subdirectories
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(f"{run_path}/traj", exist_ok=True)
    
    # Ensure outputs directory exists in the volume
    os.makedirs("/data/outputs", exist_ok=True)
    os.makedirs("/data/outputs/traj", exist_ok=True)  # For trajectory files
    
    # Add RFdiffusion to path
    os.chdir("/data/models")
    sys.path.append('/data/models/RFdiffusion')
    
    # Import necessary modules from RFdiffusion
    from inference.utils import parse_pdb
    from colabdesign.rf.utils import fix_contigs, fix_partial_contigs, fix_pdb
    from colabdesign.shared.protein import pdb_to_string
    
    # Modify options to use the run folder
    opts = [f"inference.output_prefix={run_path}/output",
            f"inference.num_designs={num_designs}"]
    
    # Sanitize inputs
    if isinstance(chains, str) and chains.strip() == "":
        chains = None
    
    # Determine symmetry type
    if symmetry in ["auto", "cyclic", "dihedral"]:
        if symmetry == "auto":
            print("Auto symmetry detection not supported in this test version")
            sym, copies = None, 1
        else:
            sym, copies = {"cyclic":(f"c{order}", order),
                        "dihedral":(f"d{order}", order*2)}[symmetry]
    else:
        symmetry = None
        sym, copies = None, 1
    
    # Parse contigs - don't split on colons here
    contigs = contigs.replace(",", " ").split()
    is_fixed, is_free = False, False
    fixed_chains = []
    
    for contig in contigs:
        for x in contig.split("/"):
            a = x.split("-")[0]
            if a[0].isalpha():
                is_fixed = True
                if a[0] not in fixed_chains:
                    fixed_chains.append(a[0])
            if a.isnumeric():
                is_free = True
                
    if len(contigs) == 0 or not is_free:
        mode = "partial"
    elif is_fixed:
        mode = "fixed"
    else:
        mode = "free"
    
    # Process PDB if provided
    pdb_str = None
    if pdb_content is not None:
        pdb_str = pdb_content
        
    # Process PDB if needed
    if mode in ["partial", "fixed"] and pdb_str:
        pdb_filename = f"{run_path}/input.pdb"
        
        # Write the PDB content to a file
        with open(pdb_filename, "w") as handle:
            handle.write(pdb_str)
            
        parsed_pdb = parse_pdb(pdb_filename)
        opts.append(f"inference.input_pdb={pdb_filename}")
        
        if mode == "partial":
            iterations = int(80 * (iterations / 200))
            opts.append(f"diffuser.partial_T={iterations}")
            contigs = fix_partial_contigs(contigs, parsed_pdb)
        else:
            opts.append(f"diffuser.T={iterations}")
            contigs = fix_contigs(contigs, parsed_pdb)
    else:
        # Fall back to free mode if no PDB
        if mode in ["partial", "fixed"]:
            mode = "free"
        opts.append(f"diffuser.T={iterations}")
        parsed_pdb = None
        contigs = fix_contigs(contigs, None)
    
    if hotspot is not None and hotspot != "":
        opts.append(f"ppi.hotspot_res=[{hotspot}]")
    
    # Setup symmetry
    if sym is not None:
        sym_opts = ["--config-name symmetry", f"inference.symmetry={sym}"]
        if add_potential:
            sym_opts += ["'potentials.guiding_potentials=[\"type:olig_contacts,weight_intra:1,weight_inter:0.1\"]'",
                       "potentials.olig_intra_all=True", "potentials.olig_inter_all=True",
                       "potentials.guide_scale=2", "potentials.guide_decay=quadratic"]
        opts = sym_opts + opts
        contigs = sum([contigs] * copies, [])
    
    opts.append(f"'contigmap.contigs=[{' '.join(contigs)}]'")
    opts += ["inference.dump_pdb=True", "inference.dump_pdb_path='/tmp'"]
    
    print("Mode:", mode)
    print("Output:", run_path)
    print("Contigs:", contigs)
    
    opts_str = " ".join(opts)
    cmd = f"cd /data/models && python RFdiffusion/run_inference.py {opts_str}"
    print(f"Running command: {cmd}")
    
    # Execute the command
    start_time = time.time()
    result = os.system(cmd)
    end_time = time.time()
    
    # Fix PDFs if necessary
    for n in range(num_designs):
        pdbs = [f"/data/outputs/traj/{folder_name}_{n}_pX0_traj.pdb",
                f"/data/outputs/traj/{folder_name}_{n}_Xt-1_traj.pdb",
                f"{run_path}_{n}.pdb"]
        for pdb_file in pdbs:
            if os.path.exists(pdb_file):
                with open(pdb_file, "r") as handle:
                    pdb_str = handle.read()
                with open(pdb_file, "w") as handle:
                    handle.write(fix_pdb(pdb_str, contigs))
    
    # After processing, ensure all files are synced and committed to the volume
    os.system("sync")
    outputs_volume.commit()
    
    # After RFdiffusion completes successfully, run MPNN
    if result == 0:
        mpnn_args = {
            "pdb": f"{run_path}/output_0.pdb",
            "loc": run_path,
            "contig": contigs,
            "copies": copies,
            "num_seqs": 8,
            "num_recycles": 1,
            "rm_aa": "C",
            "mpnn_sampling_temp": 0.1,
            "num_designs": 1
        }
        
        print("\nRunning MPNN on output structure...")
        print(f"Using PDB file: {mpnn_args['pdb']}")
        mpnn_result = run_mpnn.remote(
            mpnn_args=mpnn_args,
            initial_guess=False,
            use_multimer=False
        )
        
        return {
            "batch_path": batch_path,
            "folder_name": folder_name,
            "result": "success",
            "rfdiffusion_cmd": cmd,
            "output_path": run_path,
            "runtime_seconds": end_time - start_time,
            "contigs": contigs,
            "copies": copies,
            "mpnn_args": mpnn_args,
            "mpnn_result": mpnn_result
        }
    
    return {
        "batch_path": batch_path,
        "folder_name": folder_name,
        "result": "failed",
        "command": cmd,
        "output_path": run_path,
        "runtime_seconds": end_time - start_time,
        "contigs": contigs,
        "copies": copies,
        "mpnn_args": None
    }

@app.function(
    image=image,
    volumes={
        "/data/models": models_volume,
        "/data/outputs": outputs_volume,
    },
    gpu="A100",
    timeout=14400,
)
def run_mpnn(
    mpnn_args: dict,
    initial_guess: bool = False,
    use_multimer: bool = False,
):
    """Run ProteinMPNN on the output structure"""
    import os
    import time
    
    # First check if params are initialized
    if not os.path.isfile("/data/models/params/done.txt"):
        print("downloading AlphaFold params...")
        while not os.path.isfile("/data/models/params/done.txt"):
            time.sleep(5)
    
    # Build command line options
    opts = [
        f"--pdb={mpnn_args['pdb']}",
        f"--loc={mpnn_args['loc']}",
        f"--contig={':'.join(mpnn_args['contig']) if isinstance(mpnn_args['contig'], list) else mpnn_args['contig']}",
        f"--copies={mpnn_args['copies']}",
        f"--num_seqs={mpnn_args['num_seqs']}",
        f"--num_recycles={mpnn_args['num_recycles']}",
        f"--rm_aa={mpnn_args['rm_aa']}",
        f"--mpnn_sampling_temp={mpnn_args['mpnn_sampling_temp']}",
        f"--num_designs={mpnn_args['num_designs']}"
    ]
    
    if initial_guess:
        opts.append("--initial_guess")
    if use_multimer:
        opts.append("--use_multimer")
    
    opts_str = ' '.join(opts)
    cmd = f"python -m colabdesign.rf.designability_test {opts_str}"
    
    print(f"Running MPNN command: {cmd}")
    start_time = time.time()
    result = os.system(cmd)
    end_time = time.time()
    
    return {
        "result": "success" if result == 0 else "failed",
        "command": cmd,
        "runtime_seconds": end_time - start_time
    }

@app.local_entrypoint()
def main(
    name: str = "test",
    batch_name: str = None,  # Added batch_name parameter
    contigs: str = "100",
    pdb: str = None,
    iterations: int = 50,
    symmetry: str = "none",
    order: int = 1,
    hotspot: str = None,
    chains: str = None,
    num_designs: int = 1,
    add_potential: bool = True,
    gpu_type: str = "A100",
    timeout_hours: float = 4.0,
):
    """Modal entrypoint to run the RFdiffusion test"""
    # First make sure the volumes are initialized
    from initialize_modal import initialize_volumes
    
    print("Ensuring volumes are initialized...")
    initialize_volumes.remote()
    
    # Parse contigs list - split only on commas, preserve the rest of the structure
    contigs_list = [c.strip() for c in contigs.split(",") if c.strip()]
    print(f"Running RFdiffusion test with {len(contigs_list)} contigs:")
    for i, contig in enumerate(contigs_list):
        print(f"{i+1}. {contig}")
    
    # Configure the GPU and timeout dynamically
    timeout_seconds = int(timeout_hours * 3600)
    run_rfdiffusion_test.gpu = gpu_type
    run_rfdiffusion_test.timeout = timeout_seconds
    
    # Run all designs in parallel
    batch_name, results = run_rfdiffusion_with_local_pdb(
        name=name,
        batch_name=batch_name,
        contigs_list=contigs_list,
        pdb_path=pdb,
        iterations=iterations,
        symmetry=symmetry,
        order=order,
        hotspot=hotspot,
        chains=chains,
        add_potential=add_potential,
        num_designs=num_designs,
    )
    
    print(f"\nAll runs completed in batch: {batch_name}")
    print(f"Generated {len(results)} output folders:")
    for result in results:
        print(f"  {result['folder_name']}")
        print(f"  MPNN args: {result['mpnn_args']}")