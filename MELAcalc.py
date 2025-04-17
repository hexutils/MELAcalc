import os
import sys
import subprocess
from pathlib import Path
import argparse
import warnings
import json
import MELAweights as MW

import MELAcalc_helper as help
import Mela

def check_enum(entry, enum):
    found = False
    i = 0
    possible_value = tuple(enum.__members__.keys())
    mapping = enum.__members__
    while (not found) and (i < len(possible_value)):
        if entry.lower() == possible_value[i].lower():
            found = True
            entry = possible_value[i]
        i += 1
    if not found:
        possible_value = tuple(map(str.lower, possible_value))
        errortext = "Unknown matrix element given!"
        errortext += "\nThe following are valid matrix elements"
        errortext += "\n" + "\n".join(possible_value)
        errortext = help.print_msg_box(errortext, title="ERROR")
        raise ValueError("\n" + errortext)
    return mapping[entry]


def json_to_dict(json_file, replace_default):
    REQUIRED_ENTRIES = (
        'process',
        'matrixelement',
        'production',
        'prod',
        'dec',
        'isgen',
        'couplings',
        'computeprop'
    )
    REQUIRED_BRANCH_ENTRIES = {
        True:( #isgen=True
            "daughter_id",
            "daughter_pt",
            "daughter_eta",
            "daughter_phi",
            "daughter_mass",
            "associated_id",
            "associated_pt",
            "associated_eta",
            "associated_phi",
            "associated_mass",
            "mother_id",
            "mother_pz",
            "mother_e",
        ),
        False:( #isgen=False
            "daughter_id",
            "daughter_pt",
            "daughter_eta",
            "daughter_phi",
            "associated_pt",
            "associated_eta",
            "associated_phi",
            "associated_mass",
        )
    }

    VALID_LEXICON_BOOLEANS = (
        "useMCFMAtInput",
        "useMCFMAtOutput",
        "distinguish_HWWcouplings",
        "include_triple_quartic_guage",
        "custodial_symmetry",
        "HW_couplings_only",
        "HZ_couplings_only",
        "switch_convention"
    )
    VALID_LEXICON_PARAMETERS = (
        "input_basis",
        "Lambda_z1",
        "Lambda_w1",
        "Lambda_zgs1",
        "MZ",
        "MW",
        "sin2ThetaW",
        "alpha",
        "vev_lam",
        "delta_m",
        "delta_v",
    )
    VALID_LEXICON_COUPLINGS = (
        "ghz1",
        "ghz1_prime2",
        "ghz2",
        "ghz4",
        "ghw1",
        "ghw1_prime2",
        "ghw2",
        "ghw4",
        "ghzgs1_prime2",
        "ghzgs2",
        "ghzgs4",
        "ghgsgs2",
        "ghgsgs4",
        "ghg2",
        "ghg4",

        "dCz",
        "Czz",
        "Czbx",
        "tCzz",
        "dCw",
        "Cww",
        "Cwbx",
        "tCww",
        "Cza",
        "tCza",
        "Cabx",
        "Caa",
        "tCaa",
        "Cgg",
        "tCgg",

        "cHbx",
        "cHD",
        "cHG",
        "cHW",
        "cHB",
        "cHWB",
        "tcHG",
        "tcHW",
        "tcHB",
        "tcHWB",
    )

    with open(json_file) as json_data:
        data = json.load(json_data)

        setup_inputs = [
            {
                "process":None,
                "matrixelement":None,
                "production":None,
                "prod":None,
                "dec":None,
                "isgen":None,
                "couplings":None,
                "computeprop":None,
                "branches":None,
                "dividep":None,
                "propscheme":Mela.ResonancePropagatorScheme.FixedWidth,
                "particles":None,
                "cluster":None,
                "decaymode":Mela.CandidateDecayMode.CandidateDecay_ZZ,
                "separatewwzz":False,
                "useconstant":False,
                "match_mX":False,
                "lepton_interference":Mela.LeptonInterference.DefaultLeptonInterf,
                "replace":replace_default
            } for _ in range(len(data))] #MELA logistics, couplings, options, particles


        for n, prob_name in enumerate(data.keys()):
            current_dict = setup_inputs[n]
            current_dict["name"] = prob_name

            real_valued_couplings = set()
            for input_val in data[prob_name]:
                new_input_val = input_val.lower()

                if new_input_val not in setup_inputs[n].keys() and new_input_val != 'lexicon':
                    errortext = f"{new_input_val} is not a valid setting!"
                    errortext = help.print_msg_box(errortext, title="ERROR")
                    raise ValueError("\n" + errortext)

                settings_dict_entry = data[prob_name][input_val]
                if new_input_val == 'dividep':
                    current_dict[new_input_val] = settings_dict_entry
                    continue

                if isinstance(settings_dict_entry, str):
                    settings_dict_entry = settings_dict_entry.lower()

                if new_input_val == 'couplings':
                    if current_dict["couplings"] is not None:
                        errortext = "EFT couplings set or repeat entry!"
                        errortext += "\nYou can set either Warsaw or Higgs basis couplings!"
                        errortext += "\nNOT BOTH"
                        errortext = help.print_msg_box(errortext, title="ERROR")
                        raise ValueError("\n" + errortext)

                    for coupling in settings_dict_entry:
                        is_not_madmela_coupling = isinstance(settings_dict_entry[coupling], list)
                        if not is_not_madmela_coupling:
                            # MADMELA_COUPLING_FLAG[n] = True
                            real_valued_couplings.add(coupling)

                        # elif is_not_madmela_coupling and MADMELA_COUPLING_FLAG[n]:
                        #     errortext = f"Invalid coupling: {coupling}!"
                        #     errortext += "\nCouplings for madMELA should be real valued constants!"
                        #     errortext += "\ni.e. mdl_chwb:1"
                        #     errortext = help.print_msg_box(errortext, title="ERROR")
                        #     raise ValueError("\n" + errortext)

                        elif len(settings_dict_entry[coupling]) != 2:
                            errortext = "Length of input for " + coupling + f" is {len(settings_dict_entry[coupling])}!"
                            errortext += "\nInput for couplings should be <name>:[<real>, <imaginary>]"
                            errortext += "\ni.e. ghz1:[1,0]"
                            errortext = help.print_msg_box(errortext, title="ERROR")
                            raise ValueError("\n" + errortext)
                        del coupling

                if new_input_val == 'lexicon':
                    if current_dict["couplings"] is not None:
                        errortext = "EFT couplings set or repeat entry!"
                        errortext += "\nYou can set either Warsaw or Higgs basis couplings!"
                        errortext += "\nNOT BOTH"
                        errortext = help.print_msg_box(errortext, title="ERROR")
                        raise ValueError("\n" + errortext)

                    Lexicon_file_path = f"{os.path.dirname(os.path.abspath(__file__))}/JHUGenLexicon/JHUGenLexicon"
                    if not os.path.exists(Lexicon_file_path):
                        warningtext = "JHUGenLexicon not compiled!"
                        warningtext += "\nCompiling..."
                        warningtext = help.print_msg_box(warningtext, title="WARNING")
                        warnings.warn("\n" + warningtext)

                        pwd = os.getcwd()
                        os.chdir(f"{os.path.dirname(os.path.abspath(__file__))}/JHUGenLexicon/")
                        os.system("make")
                        os.chdir(pwd)

                    LexiconInput = f"{Lexicon_file_path} "
                    LexiconInput += "output_basis=amp_jhu alpha=7.815e-3 "
                    set_input_basis = False
                    for lex_name, val in settings_dict_entry.items():
                        found_val = False
                        comp_name = lex_name.lower()
                        if comp_name == "input_basis": set_input_basis = True
                        i = 0
                        while i < len(VALID_LEXICON_BOOLEANS) and not found_val:
                            ref_name = VALID_LEXICON_BOOLEANS[i]
                            if comp_name == ref_name.lower():
                                if not isinstance(val, bool):
                                    errortext = f"{lex_name} should be a boolean!"
                                    errortext = help.print_msg_box(errortext, title="ERROR")
                                    raise ValueError("\n" + errortext)
                                LexiconInput += f" {ref_name}={val} "
                                found_val = True
                            i += 1

                        i = 0
                        while i < len(VALID_LEXICON_PARAMETERS) and not found_val:
                            ref_name = VALID_LEXICON_PARAMETERS[i]
                            if comp_name == ref_name.lower():
                                LexiconInput += f"{lex_name}={val}"
                                found_val = True
                            i += 1

                        i = 0
                        while i < len(VALID_LEXICON_COUPLINGS) and not found_val:
                            ref_name = VALID_LEXICON_COUPLINGS[i]
                            if comp_name == ref_name.lower():
                                if not isinstance(val, list) or len(val) != 2:
                                    errortext = f"{lex_name} should be a list of <real>, <imaginary>!"
                                    errortext = help.print_msg_box(errortext, title="ERROR")
                                    raise ValueError("\n" + errortext)
                                LexiconInput += f"{lex_name}={val[0]},{val[1]} "
                                found_val = True
                            i += 1

                        if not found_val:
                            errortext = f"{lex_name} is not a valid Lexicon parameter!"
                            errortext = help.print_msg_box(errortext, title="ERROR")
                            raise KeyError("\n" + errortext)

                    if not set_input_basis:
                        errortext = "Need to set an input basis!"
                        errortext = help.print_msg_box(errortext, title="ERROR")
                        raise ValueError("\n" + errortext)
                    try:
                        LexiconOutput = subprocess.check_output(LexiconInput, shell=True, stderr=subprocess.STDOUT).decode(sys.stdout.encoding).split()
                    except subprocess.CalledProcessError as e:
                        errortext = e.output.decode(sys.stdout.encoding)
                        raise OSError(errortext)

                    title = "JHUGen Lexicon Input"
                    infotext = "\n".join(LexiconInput.split())
                    infotext += "\n" + "-"*len(title) + "\n"
                    infotext += "\n".join(LexiconOutput)
                    infotext = help.print_msg_box(infotext, title="Lexicon Conversion")
                    print(infotext)

                    settings_dict_entry = {}
                    new_input_val = 'couplings'
                    for coupl in LexiconOutput:
                        name, val = coupl.split("=")
                        real, imag = val.split(",")
                        settings_dict_entry[name] = [float(real), float(imag)]

                elif new_input_val == "particles":
                    for particle, mw_list in tuple(settings_dict_entry.items()):
                        if len(settings_dict_entry[particle]) not in (2,3):
                            errortext = f"Length of input for particle with id {particle} is {len(settings_dict_entry[particle])}!"
                            errortext += "\nInput for particles should be as <id>:[<mass>, <width>]"
                            errortext += "\nOr as <id>:[<mass>, <width>, <yukawa_mass>]"
                            errortext += "\nSet to '-1' if you want to keep the previous for any"
                            errortext += "\ni.e. 25:[100, -1] or 25:[-1, 0.01] or 3:[-1, -1, 10]"
                            errortext = help.print_msg_box(errortext, title="ERROR")
                            raise ValueError("\n" + errortext)
                        if len(mw_list) == 2:
                            mw_list += [-1] #add that the Yukawa mass is unchanged
                        settings_dict_entry[int(particle)] = tuple(mw_list)
                        del settings_dict_entry[particle] #delete the string version and keep the int
                        del particle

                elif new_input_val == 'matrixelement':
                    settings_dict_entry = check_enum(settings_dict_entry, Mela.MatrixElement)

                elif new_input_val == 'process':
                    settings_dict_entry = check_enum(settings_dict_entry, Mela.Process)

                elif new_input_val == 'production':
                    settings_dict_entry = check_enum(settings_dict_entry, Mela.Production)

                elif new_input_val == 'propscheme':
                    settings_dict_entry = check_enum(settings_dict_entry, Mela.ResonancePropagatorScheme)

                elif new_input_val == 'decaymode':
                    settings_dict_entry = check_enum(settings_dict_entry, Mela.CandidateDecayMode)

                elif new_input_val == 'lepton_interference':
                    settings_dict_entry = check_enum(settings_dict_entry, Mela.LeptonInterference)
                
                elif new_input_val == "replace":
                    if not isinstance(settings_dict_entry, bool):
                        errortext = f"The 'replace' setting must be a boolean, not {type(settings_dict_entry)}!"
                        raise TypeError("\n" + errortext)

                current_dict[new_input_val] = settings_dict_entry

            missing_entries = []
            for entry in REQUIRED_ENTRIES:
                if current_dict[entry] is None:
                    missing_entries.append(entry)

            if len(missing_entries) > 0:
                errortext = f"For prob name {prob_name}"
                errortext += "\nThe following json entries are missing:"
                errortext += "\n" + "\n".join(missing_entries)
                errortext = help.print_msg_box(errortext, title="ERROR")
                raise ValueError("\n" + errortext)
            del missing_entries

            if current_dict['branches'] is None:
                if current_dict['isgen']:
                    current_dict['branches'] = {
                        "daughter_id" : "LHEDaughterId",
                        "daughter_pt" : "LHEDaughterPt",
                        "daughter_eta" : "LHEDaughterEta",
                        "daughter_phi" : "LHEDaughterPhi",
                        "daughter_mass" : "LHEDaughterMass",
                        "associated_id" : "LHEAssociatedParticleId",
                        "associated_pt" : "LHEAssociatedParticlePt",
                        "associated_eta" : "LHEAssociatedParticleEta",
                        "associated_phi" : "LHEAssociatedParticlePhi",
                        "associated_mass" : "LHEAssociatedParticleMass",
                        "mother_id" : "LHEMotherId",
                        "mother_pz" : "LHEMotherPz",
                        "mother_e" : "LHEMotherE"
                    }
                else:
                    current_dict['branches'] = {
                        "daughter_id" : "LepLepId",
                        "daughter_pt" : "LepPt",
                        "daughter_eta" : "LepEta",
                        "daughter_phi" : "LepPhi",
                        "associated_pt" : "JetPt",
                        "associated_eta" : "JetEta",
                        "associated_phi" :  "JetPhi",
                        "associated_mass" : "JetMass"
                    }
            else:
                missing_entries = []
                for required_branch in REQUIRED_BRANCH_ENTRIES[current_dict["isgen"]]:
                    if required_branch not in current_dict['branches'].keys():
                        missing_entries.append(required_branch)

                if len(missing_entries) > 0:
                    errortext = f"For prob name {current_dict['name']} with alternative branches"
                    errortext += "\nThe following branch entries are missing:"
                    errortext += "\n" + "\n".join(missing_entries)
                    errortext = help.print_msg_box(errortext, title="ERROR")
                    raise ValueError("\n" + errortext)

        for i, config_dict in enumerate(setup_inputs):

            if len(real_valued_couplings) != 0 and config_dict['matrixelement'] != Mela.MatrixElement.MADGRAPH:
                errortext = "The following couplings are input as real-valued"
                errortext += "\nwhile not setting 'matrixelement=MADGRAPH'"
                for coupl in real_valued_couplings:
                    errortext += f"\n{coupl}"
                errortext += "\nPlease make sure you are inputting values correctly"
                errortext = help.print_msg_box(errortext, title="WARNING")
                warnings.warn("\n" + errortext)

            if config_dict['dividep'] is None: 
                #don't bother with the checks below if not relevant
                continue

            found = False
            j = 0
            while (not found) and (j < len(setup_inputs)):
                possible_division = setup_inputs[j]
                if possible_division["name"] == config_dict["dividep"]:
                    found = True
                    if j > i: #make sure that the one being divided is after the denomenator
                        setup_inputs[j], setup_inputs[i] = setup_inputs[i], setup_inputs[j]
                j += 1

            if not found:
                errortext = f"Denomenator for {config_dict['name']}, {config_dict['dividep']}, does not exist!"
                errortext = help.print_msg_box(errortext, title="ERROR")
                raise ValueError("\n" + errortext)

        return setup_inputs



def main(raw_args=None):
    parser = argparse.ArgumentParser()
    input_possibilities = parser.add_mutually_exclusive_group(required=True)
    input_possibilities.add_argument('-i', '--ifile', type=str, nargs='+', help="individual files you want weights applied to")
    input_possibilities.add_argument('-id', '--idirectory', type=str, help="An entire folder you want weights applied to")

    parser.add_argument('-o', '--outdr', type=str, required=True, help="The output folder")
    parser.add_argument('-fp', '--prefix', type=str, default="", help="Optional prefix to the output file name")
    parser.add_argument('-t', '--tBranch', type=str, default="eventTree", help="The name of the TBranch you are using")
    parser.add_argument('-ss', '--stepSize', type=int, default=100, help="The step size of iteration, in MB")

    parser.add_argument('-j', '--jsonFile', type=str, help="The JSON file containing your branch names", required=True)
    parser.add_argument('-e', '--energy', type=float, default=13, help="The center of mass energy MELA is initialized at")

    parser.add_argument('-ow', '--overwrite', action="store_true", help="Enable if you want to overwrite files in the output folder")
    parser.add_argument('-r', '--replace', action="store_true", help="Enable this if you want ALL probabilities to have replace set to 'True', instead of needing to do it in the json file")

    parser.add_argument('-v', '--verbose', choices=[0,1,2,3,4,5], type=int, default=0)
    parser.add_argument('-vl', '--verbose_local', choices=[0,1,2,3], type=int, default=0)
    parser.add_argument('-n', '--number', type=int, default=-1)
    
    args = parser.parse_args(raw_args)

    template_input = parser.format_help()

    energy_scale = args.energy

    inputfiles = args.ifile
    input_directory = args.idirectory

    #if you put in a directory instead of a set of files - this will recurse over that everything in that file
    if input_directory is not None: 
        # looking for ROOT files
        inputfiles = help.recurse_through_folder(input_directory, ".root")

    output_file_prefix = args.prefix
    outputdir = args.outdr
    step_size = args.stepSize

    json = args.jsonFile

    tbranch = args.tBranch.strip() #nasty extra spaces make us sad!
    overwrite = args.overwrite
    replace_default = args.replace
    verbosity = Mela.VerbosityLevel(args.verbose) #call enum from number
    local_verbosity = args.verbose_local
    n_events = args.number

    if not os.environ.get("LD_LIBRARY_PATH"):
        errortext = "\nPlease setup MELA first using the following command:"
        errortext += "\neval $(./setup.sh env)\n"
        errortext = help.print_msg_box(errortext, title="ERROR")
        raise os.error("\n"+errortext)

    outputdir = os.path.abspath(outputdir)
    if not outputdir.endswith("/"):
        outputdir = outputdir+"/"

    if not os.path.exists(json):
        errortext = f"File '{json}' cannot be located. Please try again with valid input.\n"
        errortext = help.print_msg_box(errortext, title="ERROR")
        raise FileNotFoundError("\n" + errortext)

    branchlist = json_to_dict(json, replace_default)

    for inputfile in inputfiles:
        if not os.path.exists(inputfile):
            errortext = f"ROOT file '{inputfile}' cannot be located. Please try again with valid input.\n"
            errortext = help.print_msg_box(errortext, title="ERROR")
            raise FileExistsError("\n" + errortext)

        outputfile = outputdir + output_file_prefix + inputfile.split('/')[-1]
        if os.path.exists(outputfile):
            if overwrite:
                warningtext =  "Overwriting "+outputfile+"\n"
                warnings.warn("\n" + help.print_msg_box(warningtext, title="WARNING"))
                os.remove(outputfile)
            else:
                errortext = f"'{outputfile}' or parts of it already exist!\n"
                errortext = help.print_msg_box(errortext, title="ERROR")
                raise FileExistsError("\n" + errortext)
        else:
            if not os.path.exists(outputdir):
                Path(outputdir).mkdir(True, True)
            print("Pre-existing output PTree not found")

        User_text = f"Input PTree is '{inputfile}'"
        User_text += f"\nInput branch is '{tbranch}'"
        User_text += f"\nOutput file is '{outputfile}'"
        print(help.print_msg_box(User_text, title="Reading user input"))
        del User_text

        calculated_probabilities = MW.addprobabilities(
            branchlist, 
            inputfile, outputfile, 
            tbranch, verbosity, 
            local_verbosity, n_events, 
            energy_scale,
            step_size
        )
        # calculated_probabilities = MW.addprobabilities(branchlist, inputfile, tbranch, verbosity, local_verbosity, n_events, energy_scale)
        # MW.dump(inputfile, tbranch, outputfile, calculated_probabilities, newTree, n_events)

if __name__ == "__main__":
    main()

