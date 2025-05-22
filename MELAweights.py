import os
import numpy as np
import uproot
import awkward as ak
import Mela
from tqdm import tqdm
from MELAcalc_helper import print_msg_box

def process_events(data_from_tree, branches, isgen, i, replacement): #if isgen just do the simpleparticlecollection
    if isgen:
        if replacement is None:
            mothers = data_from_tree[branches["mother_id"]][i][:2]
        else:
            mothers = [
                i if not replacement.get(i) else replacement.get(i) 
                for i in data_from_tree[branches["mother_id"]][i][:2]
            ]
        mother = Mela.SimpleParticleCollection_t(
            mothers,
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
    
    # associated = Mela.SimpleParticleCollection_t(
    #     data_from_tree[branches["associated_id"]][i], 
    #     data_from_tree[branches["associated_pt"]][i], 
    #     data_from_tree[branches["associated_eta"]][i], 
    #     data_from_tree[branches["associated_phi"]][i], 
    #     data_from_tree[branches["associated_mass"]][i], 
    #     True
    # )
    
    associated = None
    return (daughter, associated, mother)

def addprobabilities(
        list_of_prob_dicts, 
        infile,
        outputfile, 
        tTree, 
        verbosity, 
        local_verbose=0, 
        N_events=-1,
        energy=13, #TeV,
        step_size=100, #in MB
        replacement=None
    ):
    if not os.path.exists(infile):
        errortext = print_msg_box(infile + " does not exist!", title="ERROR")
        raise FileNotFoundError("\n" + errortext)

    m = Mela.Mela(energy, 125, verbosity)
    #Always initialize MELA at m=125 GeV
    
    f = uproot.open(infile)
    t = f[tTree]

    new_f = uproot.recreate(outputfile, compression=uproot.ZLIB(9))
    
    if (N_events < 0) or (N_events > t.num_entries):
        N_events = t.num_entries

    #get all the probability branches that you want
    possible_branches = [i["branches"] for i in list_of_prob_dicts]
    gen_status = [i["isgen"] for i in list_of_prob_dicts]
    if (
        all([i == possible_branches[0] for i in possible_branches])
        and
        all([i == gen_status[0] for i in gen_status])
    ):
        possible_branches = {"all":possible_branches[0]} #SUPER 0!
        gen_status = {"all":gen_status[0]}
    else:
        possible_branches = {i["name"]:i["branches"] for i in list_of_prob_dicts}
        gen_status = {i['name']:i["isgen"] for i in list_of_prob_dicts}
    MELA_inputs = dict()
    MELA_masses = dict()

    for event, report in uproot.iterate( #we should just pull every branch (UGH)
        t, desc=f"Looping over {tTree}",
        step_size=f"{step_size} MB",
        report=True, library="np"
    ):
        if report.tree_entry_start >= N_events:
            break
        
        N_BATCH = len(event[list(event.keys())[0]])
            
        for name, branches in possible_branches.items():
            isgen = gen_status[name]

            #replacement only happens when there is something to replace and it is a GEN event
            if (local_verbose) and (report.tree_entry_start == 0) and (replacement is not None) and isgen:    
                titular = "LHE MOTHER REPLACEMENT"
                infotext = []
                for to_replace, replace_with in replacement.items():
                    infotext.append(f"Replacing id {to_replace} with id {replace_with} in LHE Mothers")
                print(
                    print_msg_box("\n".join(infotext), title=titular)
                )

            inputs = [(branches, isgen, i, replacement) for i in range(N_BATCH)]
            MELA_inputs[name] = tuple(
                process_events(event, *i) for i in 
                tqdm(inputs, total=len(inputs), desc=f"pre-processing events for {name}")
            )
            MELA_masses[name] = [i[0].MTotal() for i in MELA_inputs[name]] #gets the sum of the 4-vectors
        
        for p, prob_dict in enumerate(list_of_prob_dicts):
            name = prob_dict["name"]
            process = prob_dict["process"]
            matrixelement = prob_dict["matrixelement"]
            production = prob_dict["production"]
            prod = prob_dict["prod"]
            dec = prob_dict["dec"]
            isPM4L = prob_dict["ispm4l"]
            isgen = prob_dict["isgen"]
            couplings = prob_dict["couplings"]
            computeprop = prob_dict["computeprop"]
            useconstant = prob_dict["useconstant"]
            match_mX = prob_dict["match_mX"]
            lepton_interference = prob_dict["lepton_interference"]
            replace = prob_dict["replace"]

            dividep = prob_dict["dividep"]
            propscheme = prob_dict["propscheme"]
            particles = prob_dict["particles"]
            if particles is None:
                particles = dict()

            decaymode = prob_dict["decaymode"]
            separatewwzz = prob_dict["separatewwzz"]

            if not replace:
                while name in event.keys():
                    for other_prob_dict in list_of_prob_dicts[p+1:]:
                        if other_prob_dict['dividep'] == name:
                            other_prob_dict['dividep'] = name + "_new"
                    name = name + "_new"
            del prob_dict

            event[name] = np.full(N_BATCH, -1, dtype=np.float64)
            if computeprop and (prod or dec):
                event[name+"_prop"] = np.full(N_BATCH, -1, dtype=np.float64)
            if isPM4L:
                event[name+"_ScaleUp"] = np.full(N_BATCH, -1, dtype=np.float64)
                event[name+"_ScaleDown"] = np.full(N_BATCH, -1, dtype=np.float64)
                event[name+"_ResUp"] = np.full(N_BATCH, -1, dtype=np.float64)
                event[name+"_ResDown"] = np.full(N_BATCH, -1, dtype=np.float64)
        
            m.setCandidateDecayMode(decaymode)
        
            if (local_verbose) and (report.tree_entry_start == 0): #This is the verbose printout area
                gigabox = []
                
                titular = "PROBABILITY BRANCH"
                infotext = "NAME = " + name
                
                for particle in particles.keys():
                    infotext += f"\nM and Ga of {particle} = {particles[particle][0]}, {particles[particle][1]}"
                
                infotext = print_msg_box(infotext, title=titular)
                
                gigabox.append(infotext)
                
                infotext = print_msg_box(process.name + ", " + matrixelement.name + ", " + production.name, title="Process, Matrix Element, Production")
                gigabox.append(infotext)
                
                infotext = print_msg_box(f"Decay mode is {decaymode.name}", title="Decay Mode")
                gigabox.append(infotext)
                
                infotext = "The following branches are used for the calculation:"
                infotext += "\n" + "\n".join(branches.keys())
                infotext = print_msg_box(infotext, title="Branches")
                gigabox.append(infotext)
                
                infotext = []
                for coupl in couplings:
                    infotext.append(coupl + f" = {couplings[coupl]}")
                infotext = "\n".join(infotext)
                infotext = print_msg_box(infotext, title="Couplings")
                gigabox.append(infotext)
                
                infotext = "prod = " + str(prod) + "\nDec = " + str(dec)
                infotext += "\nRunning "
                if prod and dec:
                    infotext += "ComputeProdDecP()"
                elif prod:
                    infotext += "ComputeProdP()"
                elif dec:
                    infotext += "ComputeP()"
                elif isPM4L: 
                    infotext += "ComputePM4l()"
                else:
                    raise ValueError("Need to select a probability calculation!")
                
                infotext = print_msg_box(infotext, title="Calculation Function")
                gigabox.append(infotext)
                
                gigabox = print_msg_box("\n".join(gigabox), title=name)
                print(gigabox)
                
                print('\n\n')
        
            if "all" in MELA_inputs.keys():
                MELA_NAME = "all"
            else:
                MELA_NAME = name
            for i in tqdm(range(N_BATCH), position=0, leave=True, desc=name):
                m.setProcess(process, matrixelement, production)
                m.setInputEvent(*MELA_inputs[MELA_NAME][i], isgen)
                if match_mX:
                    m.setMelaHiggsMassWidth(MELA_masses[MELA_NAME][i], 0.00001, 0) #set the "pole mass" to be at the summed mass
                    m.setMelaHiggsMassWidth(MELA_masses[MELA_NAME][i], 0.00001, 1)
                m.setMelaLeptonInterference(lepton_interference)
                for id, (mass, width, yukawa_mass) in particles.items():
                    if mass >= 0:
                        if id == 25:
                            m.setMelaHiggsMass(mass, 0)
                        elif id == -25:
                            m.setMelaHiggsMass(mass, 1)
                        else:
                            m.resetMass(mass, id)
                    if width >= 0:
                        if id == 25:
                            m.setMelaHiggsWidth(width, 0)
                        elif id == -25:
                            m.setMelaHiggsWidth(width, 1)
                        else:
                            m.resetWidth(width, id)
                    if yukawa_mass >= 0:
                        m.resetYukawaMass(yukawa_mass, id)

                m.differentiate_HWW_HZZ = separatewwzz
                for coupl, coupl_val in couplings.items():
                    setattr(m, coupl, coupl_val)

                if prod and dec:
                    event[name][i] = m.computeProdDecP(useconstant)
                elif prod:
                    event[name][i] = m.computeProdP(useconstant)
                elif dec:
                    event[name][i] = m.computeP(useconstant)
                elif isPM4L: 
                    event[name][i] = m.computePM4l(Mela.SuperMelaSyst.SMSyst_None)
                    event[name+"_ScaleUp"][i] = m.computePM4l(Mela.SuperMelaSyst.SMSyst_ScaleUp)
                    event[name+"_ScaleDown"][i] = m.computePM4l(Mela.SuperMelaSyst.SMSyst_ScaleDown)
                    event[name+"_ResUp"][i] = m.computePM4l(Mela.SuperMelaSyst.SMSyst_ResUp)
                    event[name+"_ResDown"][i] = m.computePM4l(Mela.SuperMelaSyst.SMSyst_ResDown)
                else:
                    raise KeyError("Need to specify either production, decay, or computeprop!")
                if computeprop and (prod or dec):
                    event[name+"_prop"][i] = m.getXPropagator(propscheme)
                elif computeprop:
                    event[name][i] = m.getXPropagator(propscheme)

                m.resetInputEvent()

            if dividep is not None:
                if dividep == name:
                    event[f"{name}_divided"] = np.full(N_BATCH, 1, dtype=float)
                else:
                    event[name] /= event[dividep]
        
        total = {key:event[key] for key in event.keys()}
        if report.tree_entry_start == 0:
            new_f[tTree] = total
        else:
            new_f[tTree].extend(total)

    new_f.close()
    f.close()
