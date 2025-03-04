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
    contigs_list=None,  # Now accepts a list of contigs
    pdb_path=None,
    iterations=50,
    symmetry="none",
    order=1,
    hotspot=None,
    chains=None,
    add_potential=True,
    num_designs=1,
    # Add MPNN parameters
    num_seqs=8,
    initial_guess=False,
    num_recycles=1,
    use_multimer=False,
    rm_aa="C",
    mpnn_sampling_temp=0.1
):
    """Run RFdiffusion with a local PDB file"""
    # Use default contig if none provided
    if contigs_list is None:
        contigs_list = ["100"]
    
    # Handle local PDB file if provided
    pdb_content = None
    if pdb_path and os.path.exists(pdb_path):
        print(f"Reading local PDB file: {pdb_path}")
        with open(pdb_path, "r") as f:
            pdb_content = f.read()
    
    # Create inputs for parallel execution
    inputs = []
    for contigs in contigs_list:
        inputs.append((
            name,
            contigs,
            pdb_content,
            iterations,
            symmetry,
            order,
            hotspot,
            chains,
            add_potential,
            num_designs,
            # Add MPNN parameters
            num_seqs,
            initial_guess,
            num_recycles,
            use_multimer,
            rm_aa,
            mpnn_sampling_temp
        ))
    
    # Run in parallel using starmap and collect all results
    print(f"Running {len(inputs)} designs in parallel...")
    # First, fully consume the generator into a list
    results = list(run_rfdiffusion_test.starmap(inputs))
    # Then create pairs with the corresponding contigs
    return [(contigs_list[i], result) for i, result in enumerate(results)]

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
    contigs="100",
    pdb_content=None,
    iterations=50,
    symmetry="none",
    order=1,
    hotspot=None,
    chains=None,
    add_potential=True,
    num_designs=1,
    # Add MPNN parameters
    num_seqs=8,
    initial_guess=False,
    num_recycles=1,
    use_multimer=False,
    rm_aa="C",
    mpnn_sampling_temp=0.1
):
    """Run RFdiffusion with the specified parameters"""
    import os
    import sys
    import random
    import string
    import time
    import json
    
    # Add RFdiffusion to path
    os.chdir("/data/models")
    sys.path.append('/data/models/RFdiffusion')
    
    # Import necessary modules from RFdiffusion
    from inference.utils import parse_pdb
    from colabdesign.rf.utils import fix_contigs, fix_partial_contigs, fix_pdb
    from colabdesign.shared.protein import pdb_to_string
    
    # Generate a unique path for outputs
    path = name
    while os.path.exists(f"/data/outputs/{path}_0.pdb"):
        path = name + "_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    
    full_path = f"/data/outputs/{path}"
    os.makedirs(full_path, exist_ok=True)
    
    # Build options for RFdiffusion
    opts = [f"inference.output_prefix={full_path}",
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
    
    # Parse contigs
    contigs = contigs.replace(",", " ").replace(":", " ").split()
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
        pdb_filename = f"{full_path}/input.pdb"
        
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
    print("Output:", full_path)
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
        pdbs = [f"/data/outputs/traj/{path}_{n}_pX0_traj.pdb",
                f"/data/outputs/traj/{path}_{n}_Xt-1_traj.pdb",
                f"{full_path}_{n}.pdb"]
        for pdb_file in pdbs:
            if os.path.exists(pdb_file):
                with open(pdb_file, "r") as handle:
                    pdb_str = handle.read()
                with open(pdb_file, "w") as handle:
                    handle.write(fix_pdb(pdb_str, contigs))
    
    # After RFdiffusion completes successfully
    if result == 0:
        # Save the output paths for later MPNN processing
        output_files = []
        for n in range(num_designs):
            pdb_file = f"{full_path}_{n}.pdb"
            if os.path.exists(pdb_file):
                output_files.append(pdb_file)
        
        return {
            "result": "success",
            "rfdiffusion_cmd": cmd,
            "output_path": full_path,
            "output_files": output_files,
            "runtime_seconds": end_time - start_time,
            "contigs": contigs,
            "copies": copies,
            "path": path
        }
    
    return {
        "result": "failed",
        "command": cmd,
        "output_path": full_path,
        "runtime_seconds": end_time - start_time,
        "contigs": contigs,
        "copies": copies,
        "path": path
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
    pdb_path: str,
    output_dir: str,
    contigs: str,
    copies: int = 1,
    num_seqs: int = 8,
    initial_guess: bool = False,
    num_recycles: int = 1,
    use_multimer: bool = False,
    rm_aa: str = "C",
    mpnn_sampling_temp: float = 0.1,
    num_designs: int = 1
):
    """Run ProteinMPNN on a previously generated structure"""
    import os
    import time
    from pathlib import Path
    
    # First check if params are initialized
    if not os.path.isfile("/data/models/params/done.txt"):
        print("Waiting for AlphaFold params...")
        while not os.path.isfile("/data/models/params/done.txt"):
            time.sleep(5)

    # Ensure paths are absolute
    pdb_path = str(Path(pdb_path).absolute())
    output_dir = str(Path(output_dir).absolute())

    mpnn_opts = [
        f"--pdb={pdb_path}",
        f"--loc={output_dir}",
        f"--contig={contigs}",
        f"--copies={copies}",
        f"--num_seqs={num_seqs}",
        f"--num_recycles={num_recycles}",
        f"--rm_aa={rm_aa}",
        f"--mpnn_sampling_temp={mpnn_sampling_temp}",
        f"--num_designs={num_designs}"
    ]
    if initial_guess:
        mpnn_opts.append("--initial_guess")
    if use_multimer:
        mpnn_opts.append("--use_multimer")
        
    # Use the installed ColabDesign package
    mpnn_cmd = f"python -m colabdesign.rf.designability_test {' '.join(mpnn_opts)}"
    print(f"Running command: {mpnn_cmd}")
    start_time = time.time()
    mpnn_result = os.system(mpnn_cmd)
    end_time = time.time()
    
    return {
        "result": "success" if mpnn_result == 0 else "failed",
        "mpnn_cmd": mpnn_cmd,
        "runtime_seconds": end_time - start_time,
        "mpnn_result": mpnn_result
    }

@app.local_entrypoint()
def main(
    name: str = "test",
    contigs: str = "100",  # Accepts a comma-separated list of contigs
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
    # Add MPNN parameters
    num_seqs: int = 8,
    initial_guess: bool = False,
    num_recycles: int = 1,
    use_multimer: bool = False,
    rm_aa: str = "C",
    mpnn_sampling_temp: float = 0.1,
):
    """Modal entrypoint to run the RFdiffusion test"""
    # First make sure the volumes are initialized
    from initialize_modal import initialize_volumes
    
    print("Ensuring volumes are initialized...")
    initialize_volumes.remote()
    
    # Parse contigs list
    contigs_list = [c.strip() for c in contigs.split(",") if c.strip()]
    print(f"Running RFdiffusion test with {len(contigs_list)} contigs: {contigs_list}")
    
    # Configure the GPU and timeout dynamically
    timeout_seconds = int(timeout_hours * 3600)
    run_rfdiffusion_test.gpu = gpu_type
    run_rfdiffusion_test.timeout = timeout_seconds
    
    # Run all contigs in parallel
    results = run_rfdiffusion_with_local_pdb(
        name=name,
        contigs_list=contigs_list,
        pdb_path=pdb,
        iterations=iterations,
        symmetry=symmetry,
        order=order,
        hotspot=hotspot,
        chains=chains,
        add_potential=add_potential,
        num_designs=num_designs,
        # Add MPNN parameters
        num_seqs=num_seqs,
        initial_guess=initial_guess,
        num_recycles=num_recycles,
        use_multimer=use_multimer,
        rm_aa=rm_aa,
        mpnn_sampling_temp=mpnn_sampling_temp,
    )
    
    print("\nAll runs completed:")
    for contigs, result in results:
        print(f"Contigs: {contigs}")
        print(f"  Result: {result['result']}")
        print(f"  Output: {result['output_path']}")
        print(f"  Runtime: {result['runtime_seconds']:.2f} seconds")
        print()

    # After getting results from RFdiffusion
    for contigs, result in results:
        if result['result'] == 'success':
            for pdb_file in result['output_files']:
                mpnn_result = run_mpnn.remote(
                    pdb_path=pdb_file,
                    output_dir=result['output_path'],
                    contigs=":".join(result['contigs']),
                    copies=result['copies'],
                    num_seqs=num_seqs,
                    initial_guess=initial_guess,
                    num_recycles=num_recycles,
                    use_multimer=use_multimer,
                    rm_aa=rm_aa,
                    mpnn_sampling_temp=mpnn_sampling_temp,
                    num_designs=num_designs
                )
                print(f"MPNN result for {pdb_file}: {mpnn_result}")