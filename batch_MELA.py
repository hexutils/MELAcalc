import os
import subprocess
import copy
import numpy as np
import json
import itertools
import uproot
import tqdm
import pickle
import Mela
from MELAcalc_helper import print_msg_box

data_from_tree = {}

def process_events(branches, isgen, i): #if isgen just do the simpleparticlecollection
    if isgen:
        mother = Mela.SimpleParticleCollection_t(
            data_from_tree[branches["mother_id"]][i][:2], 
            [0]*2, 
            [0]*2, 
            data_from_tree[branches["mother_pz"]][i][:2], 
            data_from_tree[branches["mother_e"]][i][:2], 
            False
        )
        daughter = Mela.SimpleParticleCollection_t(
            data_from_tree[branches["daughter_id"]][i], 
            data_from_tree[branches["daughter_pt"]][i], 
            data_from_tree[branches["daughter_eta"]][i], 
            data_from_tree[branches["daughter_phi"]][i], 
            [0]*len(data_from_tree[branches["daughter_phi"]][i]), 
            True
        )
    else:
        mother = None
        daughter = Mela.SimpleParticleCollection_t(
            data_from_tree[branches["daughter_id"]][i], 
            data_from_tree[branches["daughter_pt"]][i], 
            data_from_tree[branches["daughter_eta"]][i], 
            data_from_tree[branches["daughter_phi"]][i], 
            [0]*len(data_from_tree[branches["daughter_phi"]][i]), 
            True
        )
    
    associated = Mela.SimpleParticleCollection_t(
        data_from_tree[branches["associated_id"]][i], 
        data_from_tree[branches["associated_pt"]][i], 
        data_from_tree[branches["associated_eta"]][i], 
        data_from_tree[branches["associated_phi"]][i], 
        data_from_tree[branches["associated_mass"]][i], 
        True
    )
    return (daughter, associated, mother)


def pickle_events(
    branches, 
    infile, 
    outputdir,
    tTree, 
    isgen=True,
    pickle_prefix = "",
    N=-1
):
    global data_from_tree
    if not os.path.exists(infile):
        errortext = print_msg_box(infile + " does not exist!", title="ERROR")
        raise FileNotFoundError("\n" + errortext)

    f = uproot.open(infile)
    t = f[tTree]

    if N > t.num_entries or N < 0:
        N_events = t.num_entries
    else:
        N_events = N

    N_so_far = 0
    pickle_counter = 0
    pickle_files   = []

    for data_from_tree in tqdm.tqdm(t.iterate(tuple(branches.values()), step_size="10 MB", library="np"), desc="pickling events"):
        n_temp = len(data_from_tree[branches["daughter_id"]])
        if n_temp + N_so_far > N_events:
            n_temp = N_events - N_so_far
            if n_temp <= 0:
                break
        
        pickle_file = f"{outputdir}.{pickle_counter}_{pickle_prefix}_MELA_events.pkl"
        pickle_files.append(os.path.abspath(pickle_file))
        
        N_so_far += n_temp

        inputs = [(branches, isgen, i) for i in range(n_temp)]

        MELA_inputs = [process_events(*i) for i in tqdm.tqdm(inputs, total=n_temp, desc=f"Pickling file {pickle_file}")]
    
        with open(pickle_file, "wb+") as f:
            pickle_counter += 1
            pickle.dump(MELA_inputs, f)
    data_from_tree.clear()

    return pickle_files


def generate_probability_executable(
    json_file,
    original_ROOT_Tree,
    infiles,
    output_directory,
    tTree,
    outputfile,
    verbosity,
    HTcondor=True,
    executable_prefix="",
):

    original_ROOT_name = original_ROOT_Tree.split("/")[-1]

    # if not os.path.exists(infile):
    #     errortext = print_msg_box(infile + " does not exist!", title="ERROR")
    #     raise FileNotFoundError("\n" + errortext)
    
    # input_json_files = []
    # with open(json_file) as json_data:
    #     data = json.load(json_data)
    #     counter = 0
    #     os.system("rm .MELAcalc_input*.json")
    #     for item in chunks(data, n_chunks):
    #         temp_json = f".MELAcalc_input_{counter}.json"
    #         with open(temp_json, "w+") as f:
    #             json.dump(item, f, indent=4)
    #             counter += 1
    #         input_json_files.append(os.path.abspath(temp_json))

    executable = os.path.abspath(f"{executable_prefix}_batch.sh")
    with open(executable, "w+") as f:
        to_write = [
            "#!/bin/bash",
            f"cd {os.environ['MELA_LIB_PATH']}/../../ || {{ echo \"FAILED to cd\"; }}",
            "eval `scramv1 runtime -sh`",
            "eval $(./setup.sh env)",
            f"cd {os.getcwd()} || {{ echo \"FAILED to cd back\"; }}"
        ]
        output_files = []
        for i in range(len(infiles)):
            if i == 0:
                to_write +=[
                    f"if [ $1 -eq {i} ]; then",
                ]
            else:
                to_write +=[
                    f"elif [ $1 -eq {i} ]; then",
                ]
            to_write+= [
                f"\tpython3 MELAcalc_v3.py -i {original_ROOT_Tree} -p {infiles[i]}" 
                f" -v {int(verbosity)} -vl 3 -o {output_directory} -j {json_file}"
                f" -fp {i}_"
                f" || {{ echo \"FAILED input {i}\"; }}"
            ]
            output_files.append(
                output_directory + f"{i}_" + original_ROOT_Tree.split('/')[-1]
            )
            
        
        to_write += [
            f"else",
            f"\techo \"INCORRECT INPUT VALUE of $1!!!!\"",
            f"\texit 1",
            "fi"
        ]
        f.write("\n".join(to_write))
    os.system(f"chmod +x {executable}")

    if HTcondor:
        with open("MELAcalc.sub", "w+") as f:
            to_write = [
                f"executable = {executable}",
                f"arguments  = $(ProcId)",
                " ",
                f"output     = batch_logs/out_u$(i).out"
                f"error      = batch_logs/out_u$(i).err"
                f"log        = batch_logs/out_u$(i).log",
                " ",
                f"+JobFlavour = \"testmatch\"",
                f"request_memory = 20G"
                " ",
                f"queue from seq 0 {len(infiles) - 1} |"
            ]
    else:
        with open("MELAcalc.slurm", "w+") as f:
            to_write = [
                "#!/bin/bash",
                "#SBATCH --job-name=MELAcalc",
                f"#SBATCH --array=0-{len(infiles) - 1}",
                "#SBATCH --ntasks=1",
                "#SBATCH --cpus-per-task=1",
                "#SBATCH --output=batch_logs/MELA_%a.out",
                "#SBATCH --mem=20G",
                "#SBATCH --time=48:00:00",
                "#SBATCH",
                "",
                'echo "Start Job $SLURM_ARRAY_TASK_ID on $HOSTNAME"',
                "sleep 10",
                f"lhcenv -o el9 -g 12 -r \"{executable} $SLURM_ARRAY_TASK_ID\";"
            ]
            f.write("\n".join(to_write))
    
    hadd_exec = os.path.abspath(
        f"{output_directory}hadd_MELAcalc.sh"
    )
    
    with open(hadd_exec, "w+") as f:
        to_write = [
            "#!/bin/bash",
            f"hadd {output_directory + 'temp_'+original_ROOT_name} " 
            + " ".join(output_files),
            f"python3 {os.path.abspath(merge.py)} "
            f"-f {output_directory + 'temp_'+original_ROOT_name} "
            f" {original_ROOT_Tree} "
            f"-t {tTree}",
            f" {output_directory + original_ROOT_name} "
        ]
        f.write("\n".join(to_write))
        
    os.system(f"chmod +x {hadd_exec}")

