import os
import numpy as np
import uproot
import shutil
import Mela
import pickle
from tqdm import tqdm
from MELAcalc_helper import print_msg_box

def addprobabilities(
    list_of_prob_dicts, 
    original_ROOT_file,
    infile, #This is now a pickle file!
    outfile,
    tTree, 
    verbosity, 
    local_verbose=0, 
    memory_limit=int(1e8) #this is 100 MB
):
    if not os.path.exists(infile):
        errortext = print_msg_box(infile + " does not exist!", title="ERROR")
        raise FileNotFoundError("\n" + errortext)

    m = Mela.Mela(13, 125, verbosity)
    #Always initialize MELA at m=125 GeV
    
    f = uproot.open(original_ROOT_file)
    t = f[tTree]
    new_f = uproot.recreate(outfile)
    
    N_reset = memory_limit/np.dtype(np.float64).itemsize 
    #the number of times to go before 1 call to "extend"

    with open(infile, "rb") as collection:
        MELA_inputs = pickle.load(collection)
        N_events = len(MELA_inputs)
        MELA_masses = [i[0].MTotal() for i in MELA_inputs] #gets the sum of the 4-vectors

    probabilities = dict()
    
    name = None
    process = None
    matrixelement = None
    production = None
    prod = None
    dec = None
    isgen = None
    couplings = None
    computeprop = None
    propscheme = None
    particles = None
    decaymode = None
    couplings = None
    separatewwzz = None
    useconstant = None
    match_mX = None
    lepton_interference = None
    
    fields = []
    match_mX_per_setup = []
    for p, prob_dict in enumerate(list_of_prob_dicts):
        name = prob_dict["name"]
        process = prob_dict["process"]
        matrixelement = prob_dict["matrixelement"]
        production = prob_dict["production"]
        computeprop = prob_dict["computeprop"]
        prod = prob_dict["prod"]
        dec = prob_dict["dec"]
        match_mX = prob_dict["match_mX"]
        couplings = prob_dict["couplings"]
        match_mX_per_setup.append(match_mX)
        while name in t.keys():
            name = name + "_new"

        list_of_prob_dicts[p]["name"] = name #change name to the new name
        fields.append(name)
        # probabilities[name] = []
        if computeprop and (prod or dec):
            # probabilities[name+"_prop"] = []
            fields.append(name+"_prop")
        
        particles = prob_dict["particles"]
        decaymode = prob_dict["decaymode"]
        if particles is None:
            particles = dict()
        
        if local_verbose:
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
            else:
                raise ValueError("Need to select a probability calculation!")
            
            infotext = print_msg_box(infotext, title="Calculation Function")
            gigabox.append(infotext)
            
            gigabox = print_msg_box("\n".join(gigabox), title=name)
            print(gigabox)
            
            print('\n\n')

    if not any(match_mX_per_setup):
        del MELA_masses

    probabilities = {fi:[] for fi in fields}
    new_f.mktree(tTree, {fi:"float64" for fi in fields}, title="Probabilities")
    f.close()
    del fields, name, gigabox, titular, infotext, f, t, match_mX_per_setup

    for i in tqdm(range(N_events), position=0, leave=True, desc="Reweighting"):
        for p, prob_dict in enumerate(list_of_prob_dicts):
            name = prob_dict["name"]
            process = prob_dict["process"]
            matrixelement = prob_dict["matrixelement"]
            production = prob_dict["production"]
            prod = prob_dict["prod"]
            dec = prob_dict["dec"]
            isgen = prob_dict["isgen"]
            couplings = prob_dict["couplings"]
            computeprop = prob_dict["computeprop"]
            useconstant = prob_dict["useconstant"]
            match_mX = prob_dict["match_mX"]
            lepton_interference = prob_dict["lepton_interference"]
            

            propscheme = prob_dict["propscheme"]
            particles = prob_dict["particles"]
            if particles is None:
                particles = dict()

            decaymode = prob_dict["decaymode"]
            separatewwzz = prob_dict["separatewwzz"]
            
            
            m.setCandidateDecayMode(decaymode)
            m.setProcess(process, matrixelement, production)
            m.setInputEvent(*MELA_inputs[i], isgen)
            if match_mX:
                m.setMelaHiggsMassWidth(MELA_masses[i], 0.00001, 0) #set the "pole mass" to be at the summed mass
                m.setMelaHiggsMassWidth(MELA_masses[i], 0.00001, 1)
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
                probabilities[name].append(m.computeProdDecP(useconstant))
            elif prod:
                probabilities[name].append(m.computeProdP(useconstant))
            elif dec:
                probabilities[name].append(m.computeP(useconstant))
            
            
            if computeprop and (prod or dec):
                probabilities[name+"_prop"].append(m.getXPropagator(propscheme))
            elif computeprop:
                probabilities[name].append(m.getXPropagator(propscheme))

            if not (computeprop or prod or dec):
                raise KeyError("Need to specify either production, decay, or computeprop!")

            m.resetInputEvent()

        if ((i % N_reset) == 0) or (i == (N_events - 1)):
            print("resetting events")
            new_f[tTree].extend(probabilities)
            for key in probabilities.keys():
                probabilities[key] = []

    new_f.close()

    return


#         match_hmass_exactly = False
#         if "matchmh" in options.keys():
#             match_hmass_exactly = t[options['matchmh']].array(library='np')
        
#         for i in tqdm(range(N_events), position=0, leave=True, desc=prob_name):
#             if np.any(match_hmass_exactly):
#                 m.setMelaHiggsMassWidth(match_hmass_exactly[i], hwidth, 0) #Use the tree specified in matchmh to match the mass
#             else:
#                 m.setMelaHiggsMassWidth(hmass,hwidth,0)
            
#             # if z_changed:
#             for particle in particles: #particle sets the ID
#                 TUtil.SetMass(particles[particle][0], particle)
#                 TUtil.SetDecayWidth(particles[particle][1], particle)
            
#             m.setRenFacScaleMode(scale_scheme, scale_scheme, ren_scale, fac_scale)
            
#             # Setup event information depending on RECO or LHE level #
#             m.setInputEvent(lepton_list[i], jets_list[i], mothers_list[i], inputEventNum)
            
#             m.setProcess(MELA_process, MELA_matrix_element, MELA_production)
            
#             for coupl in couplings:
#                 if i == 0 and coupl not in dir(m) and not special_cases(coupl):
#                     errortext = "Coupling " + coupl + " does not exist!"
#                     raise ModuleNotFoundError("\n" + print_msg_box(errortext, title="ERROR"))
                    
#                 if 'ghz' or 'ghw' in coupl:
#                     m.differentiate_HWW_HZZ = True
                
#                 if not special_cases(coupl):
#                     setattr(m, coupl, couplings[coupl])
#                 ###### NOW BEGINS THE SPECIAL CASES ######
#                 elif 'ghv' in coupl:
#                     coupl_1 = coupl.replace('v', 'z')
#                     coupl_2 = coupl.replace('v', 'w')
                    
#                     if coupl_1 not in dir(m):
#                         errortext = "Coupling " + coupl_1 + " does not exist"
#                         raise ModuleNotFoundError("\n" + print_msg_box(errortext, title="ERROR"))
                    
#                     if coupl_2 not in dir(m):
#                         errortext = "Coupling " + coupl_2 + " does not exist"
#                         raise ModuleNotFoundError("\n" + print_msg_box(errortext, title="ERROR"))
                    
#                     if local_verbose > 1 and i == 0:
#                         print("Special case " + coupl + " -> " + coupl_1 + " and " + coupl_2)
                        
#                     setattr(m, coupl_1, couplings[coupl])
#                     setattr(m, coupl_2, couplings[coupl])
#                 else:
#                     errortext = coupl + " Is an unhandled special case!"
#                     raise ValueError("\n" + print_msg_box(errortext, title="ERROR")) #handles the "special cases"
            
#             if 'bsm' in options.keys() and options['bsm'].lower() == "ac": #jerry-rigged BSM calculation
#                 gha2_cpl       = couplings["gha2"]
#                 ghz2_cpl       = couplings["ghz2"]
#                 ghza2_cpl      = couplings["ghza2"]
#                 ghz1prime2_cpl = couplings["ghz1prime2"]
#                 gha4_cpl       = couplings["gha4"]
#                 ghz4_cpl       = couplings["ghz4"]
#                 ghza4_cpl      = couplings["ghza4"]
                
#                 sin2thetaW = 0.23119
#                 mZ = 91.1876 # [GeV]
#                 lambda_Z1 = 10*1000 # [TeV] -> [GeV]
            
#                 m.dV_A = 1 + (gha2_cpl - ghz2_cpl)*(1-sin2thetaW) + ghza2_cpl*(np.sqrt((1-sin2thetaW)/sin2thetaW) - 2*np.sqrt(sin2thetaW*(1-sin2thetaW)))
#                 m.dP_A = 1
#                 m.dM_A = 1
#                 m.dFour_A = (gha4_cpl - ghz4_cpl)*(1-sin2thetaW) + ghza4_cpl*(np.sqrt((1-sin2thetaW)/sin2thetaW) - 2*np.sqrt(sin2thetaW*(1-sin2thetaW)))

#                 m.dV_Z = 1 - 2*((sin2thetaW*(1-sin2thetaW))/(1-sin2thetaW-sin2thetaW))*(gha2_cpl-ghz2_cpl) - 2*np.sqrt(sin2thetaW*(1-sin2thetaW))*ghza2_cpl - (mZ**2)/(2*(1-sin2thetaW-sin2thetaW))*(ghz1prime2_cpl/lambda_Z1**2)
#                 m.dP_Z = 1 - sin2thetaW/(1-sin2thetaW-sin2thetaW)*(gha2_cpl-ghz2_cpl) - np.sqrt(sin2thetaW/(1-sin2thetaW))*ghza2_cpl - (mZ**2)/(2*(1-sin2thetaW-sin2thetaW))*(ghz1prime2_cpl/lambda_Z1**2)
#                 m.dM_Z = m.dP_Z
#                 m.dFour_Z = -np.sqrt(sin2thetaW/(1-sin2thetaW))*m.dFour_A

#                 m.dAAWpWm = 1
#                 m.dZAWpWm = m.dP_Z
#                 m.dZZWpWm = 2*m.dP_Z - 1
            
#             if calc_decay and calc_production:
#                 probabilities[prob_name][i] = np.float64(m.computeProdDecP())
#             elif calc_decay:
#                 probabilities[prob_name][i] = np.float64(m.computeP())
#             elif calc_production:
#                 probabilities[prob_name][i] = np.float64(m.computeProdP())
#             else:
#                 raise KeyError("Need to specify either production or decay!")
            
#             if local_verbose > 2:
#                 print(f"Probability {prob_name} for iteration {i} = {probabilities[prob_name][i]:.5e}")
            
#             m.resetInputEvent()
    
#         if 'dividep' in options.keys():
#             if local_verbose > 1:
#                 print("Dividing probability", prob_name, "by", options['dividep'])
#                 old = probabilities[prob_name]
            
#             divisor_name = options['dividep']
            
#             if divisor_name not in probabilities.keys():
#                 errortext = f"Unable to divide {prob_name} by {divisor_name}"
#                 errortext += f"\nProbability {divisor_name} should be calculated first!"
#                 raise KeyError("\n" + errortext, title="ERROR")
            
#             elif divisor_name == prob_name:
#                 probabilities[prob_name + "_scaled"] = np.ones(probabilities[prob_name].shape, dtype=np.float64)
#             else:
#                 probabilities[prob_name + "_scaled"] = probabilities[prob_name].copy()/probabilities[divisor_name]
            
#             if local_verbose > 1:
#                 new = probabilities[prob_name]
#                 print(f"{'old':^9} {'new':^9}")
#                 print(*[(i,j) for i,j in zip(old, new)], sep='\n')
    
#     [calculate_probabilities(*i) for i in list_of_prob_dicts]
#     return probabilities


# def dump(infile, tTree, outfile, probabilities, newTree="", N_events=-1):
#     if newTree != "":
#         shutil.copy2(infile, outfile)
#         newf = uproot.update(outfile)
#         newf[newTree] = probabilities
#         newf.close()
#         return

#     f = ROOT.TFile(infile)
#     t = f.Get(tTree)
#     newf = ROOT.TFile(outfile, "RECREATE")
#     newt = t.CloneTree(0)
    
#     if (N_events < 0) or (N_events > len(probabilities[list(probabilities.keys())[0]])):
#         N_events = t.GetEntries()
    
#     root_input = [None]*len(probabilities)
#     for n, prob in enumerate(probabilities.keys()):
#         root_input[n] = np.array([0.], dtype=float)
#         newt.Branch(prob, root_input[n], prob+"/D")
#     for i in tqdm(range(N_events), desc="Dumping"):
#         for n, prob in enumerate(probabilities):
#             t.GetEntry(i)
#             root_input[n][0] = probabilities[prob][i]
#         newt.Fill()
#     newf.Write()
#     newf.Close()
#     f.Close()
#     return
