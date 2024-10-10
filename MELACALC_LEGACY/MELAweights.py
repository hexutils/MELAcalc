# from ..JHUGenMELA.MELA.python.mela import Mela, TVar, SimpleParticle_t, SimpleParticleCollection_t
import ROOT, os, sys, re, numpy as np
sys.path.append('../JHUGenMELA/MELA/python/')
from mela import Mela, TVar, SimpleParticle_t, SimpleParticleCollection_t, TUtil
from tqdm import tqdm

def tlv(pt, eta, phi, m):
  result = ROOT.TLorentzVector()
  result.SetPtEtaPhiM(pt, eta, phi, m)
  return result

def parse_prob(probability):
#  print(probability)
  ## Functionality only tested for ggH mode ##
  parsed_dict={"Process":None,"ProdMode":None,"MatrixElement":None,"coupl_dict":{},"isReco":None,"Prod":None,"Dec":None,"JES":None,"JEC":None,"JER":None,
               "BSM":None, "SPIN":0}
  # Sort whether to use Jet systematics #
  if "JES" in probability: 
    if "Up" in probability:
      parsed_dict["JES"] = "Up"
    elif "Down" in probability:
      parsed_dict["JES"] = "Down"
    else:
      raise ValueError("Invalid JES option")
  if "JEC" in probability:
    if "Nominal" in probability: 
      parsed_dict["JEC"] = "Nominal"
    else:
      raise ValueError("Invalid JEC option")
  if "JER" in probability:
    if "Up" in probability:
      parsed_dict["JER"] = "Up"
    elif "Down" in probability:
      parsed_dict["JER"] = "Down"
    else:
      raise ValueError("Invalid JER option")
  if [parsed_dict["JES"], parsed_dict["JEC"], parsed_dict["JER"]].count(None) < 2:
    raise ValueError("Invalid combination of JES,JEC,JER!")
  # Sort Process over #
  if "SIG" in probability:
    parsed_dict["Process"] = "SIG"
  elif "BKG" in probability:
    parsed_dict["Process"] = "BKG"
  elif "BSI" in probability:
    parsed_dict["Process"] = "BSI"
  #Sort Reco Or Not#
  if "_Gen" in probability:
    parsed_dict["isReco"] = False
  else:
    parsed_dict["isReco"] = True
  # Sort Production Mode #
  if "GG" in probability:
    parsed_dict["ProdMode"] = "GG"
    parsed_dict["Prod"] = False
    parsed_dict["Dec"] = True
  elif "QQ" in probability:
    parsed_dict["ProdMode"] = "QQ"
    parsed_dict["Prod"] = False
    parsed_dict["Dec"] = True
  elif "LepZH" in probability:
    parsed_dict["ProdMode"] = "LepZH"
    parsed_dict["Prod"] = True
    parsed_dict["Dec"] = False
  elif "HadZH" in probability:
    parsed_dict["ProdMode"] = "HadZH"
    if "JHUGen" in probability:
      parsed_dict["Prod"] = True
      parsed_dict["Dec"] = False
    elif "MCFM" in probability:
      parsed_dict["Prod"] = True
      parsed_dict["Dec"] = True
    else:
      raise ValueError("Choose correct Matrix element for HadZH")
  elif "LepWH" in probability:
    parsed_dict["ProdMode"] = "LepWH"
    parsed_dict["Prod"] = True
    parsed_dict["Dec"] = False
  elif "HadWH" in probability:
    parsed_dict["ProdMode"] = "HadWH"
    if "JHUGen" in probability:
      parsed_dict["Prod"] = True
      parsed_dict["Dec"] = False
    elif "MCFM" in probability:
      parsed_dict["Prod"] = True
      parsed_dict["Dec"] = True
    else:
      raise ValueError("Choose correct Matrix element for HadWH")
  elif "JJEW" in probability:
    parsed_dict["ProdMode"] = "JJEW"
    parsed_dict["Prod"] = True
    parsed_dict["Dec"] = True
  elif "JJVBF" in probability:
    parsed_dict["ProdMode"] = "JJVBF"
    if "JHUGen" in probability:
      parsed_dict["Prod"] = True
      parsed_dict["Dec"] = False
    elif "MCFM" in probability:
      parsed_dict["Prod"] = True
      parsed_dict["Dec"] = True
    else:
      raise ValueError("Choose correct Matrix element for JJVBF")
  elif "JVBF" in probability:
    parsed_dict["ProdMode"] = "JJEWQCD"
    parsed_dict["Prod"] = True
    parsed_dict["Dec"] = False
  elif "JJQCD" in probability:
    parsed_dict["ProdMode"] = "JJQCD"
    if "JHUGen" in probability:
      parsed_dict["Prod"] = True
      parsed_dict["Dec"] = False
    elif "MCFM" in probability:
      parsed_dict["Prod"] = True
      parsed_dict["Dec"] = True
    else:
      raise ValueError("Choose correct Matrix element for JJQCD")
  elif "JQCD" in probability:
    parsed_dict["ProdMode"] = "JQCD"
    parsed_dict["Prod"] = True
    parsed_dict["Dec"] = False
  elif "ttH" in probability:
    parsed_dict["ProdMode"] = "ttH"
    parsed_dict["Prod"] = True
    parsed_dict["Dec"] = False
  elif "bbH" in probability:
    parsed_dict["ProdMode"] = "bbH"
    parsed_dict["Prod"] = True
    parsed_dict["Dec"] = False
  elif "IND" in probability:
    parsed_dict["ProdMode"] = "IND"
    parsed_dict["Prod"] = False
    parsed_dict["Dec"] = True
  # Sort MatrixElement #
  if "JHUGen" in probability:
    parsed_dict["MatrixElement"] = "JHUGen"
  elif "MCFM" in probability:
    parsed_dict["MatrixElement"] = "MCFM"
  # Raise Anomalous Coupling Flag #
  if "AC" in probability:
    parsed_dict["BSM"] = "AC"
  # Sort Couplings #
  coupling_names = re.findall(r"[a-zA-Z0-9]+_(?:m)*[0-9]?(?:p)*?[0-9]", probability)
  coupling_value_tuples = re.findall("[a-zA-Z0-9]+_((?:m)*[0-9]?(?:p)*?[0-9]+)(?:[eE]([+-]?\d+))?",probability)
  for i in range(len(coupling_names)):
    coupling = coupling_names[i].split("_")[0]
    if coupling_value_tuples[i][1] == '':
      if "p" in coupling_value_tuples[i][0]:
        if "m" in coupling_value_tuples[i][0]:
          value = -float(coupling_value_tuples[i][0].split("m")[1].split("p")[0] + "." + coupling_value_tuples[i][0].split("m")[1].split("p")[1])
        else:
          value = float(coupling_value_tuples[i][0].split("p")[0] + "." + coupling_value_tuples[i][0].split("p")[1])
      else:
        if "m" in coupling_value_tuples[i][0]:
          value = -float(coupling_value_tuples[i][0].split("m")[1])
        else:
          value = float(coupling_value_tuples[i][0])
    else:
      if "p" in coupling_value_tuples[i][0]:
        if "m" in coupling_value_tuples[i][0]:
          value = -float(coupling_value_tuples[i][0].split("m")[1].split("p")[0] + "." + coupling_value_tuples[i][0].split("m")[1].split("p")[1])* 10 ** float(coupling_value_tuples[i][1])
        else:
          value = float(coupling_value_tuples[i][0].split("p")[0] + "." + coupling_value_tuples[i][0].split("p")[1]) * 10 ** float(coupling_value_tuples[i][1])
      else:
        if "m" in coupling_value_tuples[i][0]:
          value = -float(coupling_value_tuples[i][0]).split("m")[1] * 10 ** float(coupling_value_tuples[i][1])
        else:
          value = float(coupling_value_tuples[i][0]) * 10 ** float(coupling_value_tuples[i][1])
    parsed_dict["coupl_dict"][coupling] = value
  
  if "spin1" in probability:
    parsed_dict['SPIN'] = 1
  elif "spin2" in probability:
    parsed_dict['SPIN'] = 2
  
  if parsed_dict["ProdMode"] == None:
    raise ValueError("Coupling does not have a valid Production Mode")
  if parsed_dict["MatrixElement"] == None:
    raise ValueError("Coupling does not have a valid MatrixElement")
  if parsed_dict["Process"] == None:
    print('\n',parsed_dict,'\n')
    raise ValueError("Coupling does not have a valid Process")
  # print (parsed_dict)
  return parsed_dict

def exportPath():
  os.system("export LD_LIBRARY_PATH=../JHUGenMELA/MELA/data/$SCRAM_ARCH/:${LD_LIBRARY_PATH}")

def addprobabilities(infile,outfile,probabilities,TreePath, 
                    hasJets=False, 
                    SampleHypothesisMCFM=None, SampleHypothesisJHUGen=None,
                    higgsMass=125, couplings=[None], verbosity=0, 
                    **kwargs):
  
  HasMCFMSampleHypothesis = False
  HasJHUGenSampleHypothesis = False
  SampleHypothesisMCFM = kwargs.get('SampleHypothesisMCFM', None)
  SampleHypothesisJHUGen = kwargs.get('SampleHypothesisJHUGen', None)
  
  # higgsMass = 125
  
    
  
  if SampleHypothesisMCFM != None:
    HasMCFMSampleHypothesis = True
  if SampleHypothesisJHUGen != None:
    HasJHUGenSampleHypothesis = True
  assert os.path.exists(infile)
  #assert not os.path.exists(outfile)

  if os.path.exists(outfile):
    for i in range(1000):
      if not os.path.exists(outfile.replace(".root", "_{}.root".format(i))): 
        outfile = outfile.replace(".root", "_{}.root".format(i))
        break

  #print(outfile)

  exportPath()
  
  m = Mela(13, higgsMass, verbosity)#TVar.DEBUG_MECHECK) #<- this is the debugger! 
  #Use it as another argument if you'd like to debug code
  #Always initialize MELA at m=125 GeV
  print('\nThis is the mass of the "Higgs":', higgsMass)
  
  f = ROOT.TFile(infile)
  t = f.Get(TreePath)
  print(infile, TreePath, t)
  try:
    newf = ROOT.TFile(outfile, "RECREATE")
    newt = t.CloneTree(0)
    print(newt)
    newtbranches = newt.GetListOfBranches()
    probdict = {}

  
    for prob in probabilities: 
      probname = prob
      #print(probname)
      if probname in newtbranches:
        for i in range(1000):
            probname = prob+"_{}".format(i)
            #print(probname)
            if not probname in newtbranches: break
      #print(probname, prob)
      probdict[prob]=np.array([0],dtype=np.float32)
      newt.Branch(probname,probdict[prob],probname+"/F")
      #print(probdict)
    if HasMCFMSampleHypothesis:
      probdict[SampleHypothesisMCFM]=np.array([0],dtype=np.float32)
    if HasJHUGenSampleHypothesis:
      probdict[SampleHypothesisJHUGen]=np.array([0],dtype=np.float32)
    #print(probdict)

    #print(parse_prob(prob))
    
    #sys.exit()

    for i, entry in enumerate(tqdm(t, position=0, leave=True), start=1):
      #####################################################
      # RECO probabilities, for reweighting Discriminants #
      #####################################################
      #once per event
      for prob in probdict:
        # print()
        # Parse Relevant Information from Coupling String #
        parsed_prob_dict = parse_prob(prob)
        Process = None
        JetPt = None
        MatrixElement = None
        Production = None
        ns = {'t':entry,'JetPt':JetPt,'Process':Process,'MatrixElement':MatrixElement,'Production':Production,'TVar':TVar}
        # Setup the correct Jet Scales etc #
        JetPtExec='JetPt=t.'
        if parsed_prob_dict["JES"] == "Up":
          JetPtExec+='JetPt_JESUp'
        elif parsed_prob_dict["JES"] == "Down":
          JetPtExec+='JetPt_JESDown'
        elif parsed_prob_dict["JER"] == "Up":
          JetPtExec+='JetPt_JERUp'
        elif parsed_prob_dict["JER"] == "Down":
          JetPtExec+='JetPt_JESDown'
        else:
          JetPtExec+="JetPt"
        try:
          exec(JetPtExec,ns)
        except:
          print("Error in choosing JetPt Uncertainty")
        # Setup event information depending on RECO or LHE level #
        if parsed_prob_dict["isReco"]:
          leptons = SimpleParticleCollection_t(SimpleParticle_t(pid, tlv(pt, eta, phi, 0)) for pid, pt, eta, phi in zip(t.LepLepId, t.LepPt, t.LepEta, t.LepPhi))
          jets = SimpleParticleCollection_t(SimpleParticle_t(0, tlv(pt, eta, phi, m)) for pt, eta, phi, m in zip(t.JetPt, t.JetEta, t.JetPhi, t.JetMass))
          mothers = 0
          m.setInputEvent(leptons, jets, mothers, 0)
        else:
          leptons = SimpleParticleCollection_t(SimpleParticle_t(pid, tlv(pt, eta, phi, m)) for pid, pt, eta, phi, m in zip(t.LHEDaughterId, t.LHEDaughterPt, t.LHEDaughterEta, t.LHEDaughterPhi, t.LHEDaughterMass))
          jets = SimpleParticleCollection_t(SimpleParticle_t(pid, tlv(pt, eta, phi, m)) for pid, pt, eta, phi, m in zip(t.LHEAssociatedParticleId, t.LHEAssociatedParticlePt, t.LHEAssociatedParticleEta, t.LHEAssociatedParticlePhi, t.LHEAssociatedParticleMass))
          mothers = SimpleParticleCollection_t(SimpleParticle_t(pid, ROOT.TLorentzVector(0, 0, pz, e)) for pid, pz, e in zip(t.LHEMotherId, t.LHEMotherPz, t.LHEMotherE))
          m.setInputEvent(leptons, jets, mothers, 1) 
        # Sort the MatrixElement #
        MatrixExec = "MatrixElement = TVar."+parsed_prob_dict["MatrixElement"]
        exec(MatrixExec,ns)
        try:
          exec(MatrixExec,ns)
        except:
          print("Choose Valid Matrix Element")
        # Sort the production Mode and Generator #
        ProdExec = "Production = TVar."
        if parsed_prob_dict["ProdMode"] == "GG":
          ProdExec+="ZZGG"
        elif parsed_prob_dict["ProdMode"] == "QQ":
          ProdExec+="ZZQQB"
        elif parsed_prob_dict["ProdMode"] == "IND":
          ProdExec+="ZZINDEPENDENT"
        else:
          ProdExec+=parsed_prob_dict["ProdMode"]
        #print("\n\n", ProdExec, "\n\n")
        try:
          #print("\n\n", ProdExec, "\n\n")
          exec(ProdExec,ns)
        except:
          print("Choose Valid Production Mode")
        #Sort out the process
        ProcExec = "Process = TVar."

        if parsed_prob_dict["MatrixElement"] == "MCFM":
            '''
            ProcExec += "bkgZZ_SMHiggs"
            '''
            if parsed_prob_dict["ProdMode"] == "GG":
                if parsed_prob_dict["Process"] == "BKG":
                    if len(parsed_prob_dict["coupl_dict"]) == 0:
                        ProcExec += "bkgZZ"
                    else:
                        ProcExec += "bkgZZ"
                elif parsed_prob_dict["Process"] == "BSI":
                    if len(parsed_prob_dict["coupl_dict"]) == 0:
                        ProcExec += "bkgZZ_SMHiggs"
                    else:
                        ProcExec += "bkgZZ_SMHiggs"
                elif parsed_prob_dict["Process"] == "SIG":
                    if len(parsed_prob_dict["coupl_dict"]) == 0:
                        ProcExec += "HSMHiggs"
                    else:
                        ProcExec += "HSMHiggs"
                else:
                    ProcExec += "SelfDefine_spin0"
            elif parsed_prob_dict["ProdMode"] == "JVBF":
                if parsed_prob_dict["Process"] == "BKG":
                    if len(parsed_prob_dict["coupl_dict"]) == 0:
                        ProcExec += "bkgZZ"
                    else:
                        ProcExec += "bkgZZ"
                elif parsed_prob_dict["Process"] == "BSI":
                    if len(parsed_prob_dict["coupl_dict"]) == 0:
                        ProcExec += "bkgZZ_SMHiggs"
                    else:
                        ProcExec += "bkgZZ_SMHiggs"
                elif parsed_prob_dict["Process"] == "SIG":
                    if len(parsed_prob_dict["coupl_dict"]) == 0:
                        ProcExec += "HSMHiggs"
                    else:
                        ProcExec += "HSMHiggs"
                else:
                    ProcExec += "SelfDefine_spin0"
            elif parsed_prob_dict["ProdMode"] == "JJEW":
                if parsed_prob_dict["Process"] == "BKG":
                    if len(parsed_prob_dict["coupl_dict"]) == 0:
                        ProcExec += "bkgZZ"
                    else:
                        ProcExec += "bkgZZ"
                elif parsed_prob_dict["Process"] == "BSI":
                    if len(parsed_prob_dict["coupl_dict"]) == 0:
                        ProcExec += "bkgZZ_SMHiggs"
                    else:
                        ProcExec += "bkgZZ_SMHiggs"
                elif parsed_prob_dict["Process"] == "SIG":
                    if len(parsed_prob_dict["coupl_dict"]) == 0:
                        ProcExec += "HSMHiggs"
                    else:
                        ProcExec += "HSMHiggs"
                else:
                    ProcExec += "SelfDefine_spin0"
            DivideP = True 
            
        elif parsed_prob_dict["MatrixElement"] == "JHUGen":
            if parsed_prob_dict['SPIN'] == 0:
              ProcExec += "SelfDefine_spin0"
            elif parsed_prob_dict['SPIN'] == 1:
              ProcExec += "SelfDefine_spin1"
            elif parsed_prob_dict['SPIN'] == 2:
              ProcExec += "SelfDefine_spin2"
            else:
              raise "Invalid Spin type!"
        
        try:
            exec(ProcExec,ns)
        except:
            print("Current Process Not Supported")
        
        # print("\n\n", parsed_prob_dict, "\n\n")

        # print("\n\n", parsed_prob_dict, "\n\n")

        # print("\n\n", ns, "\n\n")

        # print("\n\n", ns['Process'],ns['MatrixElement'],ns['Production'], "\n\n")

        # print("\n\n", ns, "\n\n")

        # print("\n\n===========================================================================================================================================================\n\n")


        #print(parsed_prob_dict['coupl_dict'])
        m.setProcess(ns['Process'],ns['MatrixElement'],ns['Production'])
        
        # print(parsed_prob_dict)
        bkg_prob_dict = {"gha2":0, "ghz2":0, "ghza2":0, "gha4":0, "ghz4":0, "ghza4":0, "ghz1prime2":0} 
        if parsed_prob_dict["Process"] == "BKG":
            # Sort Couplings
            for key in parsed_prob_dict['coupl_dict']:
              if key == "ghz2":
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghz4":
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "gha2":
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "gha4":
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghza2":
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghza4":
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghz1prime2":
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghv2":
                bkg_prob_dict["ghz2"] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghv4":
                bkg_prob_dict["ghz4"] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghv1prime2":
                bkg_prob_dict["ghz1prime2"] = parsed_prob_dict['coupl_dict'][key]
        else:
            # Sort Couplings
            for key in parsed_prob_dict['coupl_dict']:
              if 'ghw' in key or 'ghz' in key:
                m.differentiate_HWW_HZZ = True
              
              if key == "ghg2":
                m.ghg2 = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghg4":
                m.ghg4 = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghz1":
                m.ghz1 = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghz2":
                m.ghz2 = parsed_prob_dict['coupl_dict'][key]
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghz4":
                m.ghz4 = parsed_prob_dict['coupl_dict'][key]
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghw1":
                m.ghw1 = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghw2":
                m.ghw2 = parsed_prob_dict['coupl_dict'][key]
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghw4":
                m.ghw4 = parsed_prob_dict['coupl_dict'][key]
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "gha2":
                m.ghgsgs2 = parsed_prob_dict['coupl_dict'][key]
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "gha4":
                m.ghgsgs4 = parsed_prob_dict['coupl_dict'][key]
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghza2":
                m.ghzgs2 = parsed_prob_dict['coupl_dict'][key]
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghza4":
                m.ghzgs4 = parsed_prob_dict['coupl_dict'][key]
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "kappaTopBot":
                m.kappa_top = parsed_prob_dict['coupl_dict'][key]
                m.kappa_top_tilde = 0
                m.kappa_bot = parsed_prob_dict['coupl_dict'][key]
                m.kappa_bot_tilde = 0
              elif key == "kappa":
                m.kappa = parsed_prob_dict['coupl_dict'][key]
              elif key == "kappatilde":
                m.kappa_tilde = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghza1prime2":
                m.ghzgs1_prime2 = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghz1prime2":
                m.ghz1_prime2 = parsed_prob_dict['coupl_dict'][key]
                bkg_prob_dict[key] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghv1":
                m.ghz1 = parsed_prob_dict['coupl_dict'][key]
                m.ghw1 = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghv2":
                m.ghz2 = parsed_prob_dict['coupl_dict'][key]
                m.ghw2 = parsed_prob_dict['coupl_dict'][key]
                bkg_prob_dict["ghz2"] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghv4":
                m.ghz4 = parsed_prob_dict['coupl_dict'][key]
                m.ghw4 = parsed_prob_dict['coupl_dict'][key]
                bkg_prob_dict["ghz4"] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghv1prime2":
                m.ghz1_prime2 = parsed_prob_dict['coupl_dict'][key]
                m.ghw1_prime2 = parsed_prob_dict['coupl_dict'][key]
                bkg_prob_dict["ghz1prime2"] = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghzpzp1":
                m.ghzpzp1 = parsed_prob_dict['coupl_dict'][key]
              elif key == "ghzpzp4":
                m.ghzpzp4 = parsed_prob_dict['coupl_dict'][key]
              elif key == "rez":
                m.reZ = parsed_prob_dict['coupl_dict'][key]
              elif key == "lez":
                m.leZ = parsed_prob_dict['coupl_dict'][key]
              elif key == 'rezpmu':
                m.ezp_Mu_right = parsed_prob_dict['coupl_dict'][key]
              elif key == 'lezpmu':
                m.ezp_Mu_left = parsed_prob_dict['coupl_dict'][key]
              elif key == "mz":
                # m.resetMass(parsed_prob_dict['coupl_dict'][key], 12) #For some stupid reason Ulas made the Z=12
                TUtil.SetMass(parsed_prob_dict['coupl_dict'][key], 23)
              elif key == "gaz":
                # m.resetWidth(parsed_prob_dict['coupl_dict'][key], 12)
                TUtil.SetDecayWidth(parsed_prob_dict['coupl_dict'][key], 23)
              elif key == 'mzp':
                m.M_Zprime = parsed_prob_dict['coupl_dict'][key]
              elif key == 'gazp':
                m.Ga_Zprime = parsed_prob_dict['coupl_dict'][key]
              elif key == 'zprimeqqleft':
                m.zprime_qq_left = parsed_prob_dict['coupl_dict'][key]
              elif key == 'zprimeqqright':
                m.zprime_qq_right = parsed_prob_dict['coupl_dict'][key]
              elif key == "zprimezz1":
                m.zprime_zz_1 = parsed_prob_dict['coupl_dict'][key]
              elif key == "zprimezz2":
                m.zprime_zz_2 = parsed_prob_dict['coupl_dict'][key]
              elif key == 'a1':
                m.a1 = parsed_prob_dict['coupl_dict'][key]
              elif key == 'b1':
                m.b1 = parsed_prob_dict['coupl_dict'][key]
              elif key == 'b5':
                m.b5 = parsed_prob_dict['coupl_dict'][key]
              else:
                print(str(key))
                print(parsed_prob_dict['coupl_dict'])
                raise ValueError(str(key) + " is not a supported coupling!")
        
        # setting gauge boson self couplings for BKG prob calc
        # only includes Z couplings
        if parsed_prob_dict["BSM"] == "AC":
          gha2_cpl = bkg_prob_dict["gha2"]
          ghz2_cpl = bkg_prob_dict["ghz2"]
          ghza2_cpl = bkg_prob_dict["ghza2"]
          ghz1prime2_cpl = bkg_prob_dict["ghz1prime2"]
          gha4_cpl = bkg_prob_dict["gha4"]
          ghz4_cpl = bkg_prob_dict["ghz4"]
          ghza4_cpl = bkg_prob_dict["ghza4"]
          
          sin2thetaW = 0.23119
          mZ = 91.1876 # [GeV]
          lambda_Z1 = 10*1000 # [TeV] -> [GeV]
        
          m.dV_A = 1 + (gha2_cpl - ghz2_cpl)*(1-sin2thetaW) + ghza2_cpl*(np.sqrt((1-sin2thetaW)/sin2thetaW) - 2*np.sqrt(sin2thetaW*(1-sin2thetaW)))
          m.dP_A = 1
          m.dM_A = 1
          m.dFour_A = (gha4_cpl - ghz4_cpl)*(1-sin2thetaW) + ghza4_cpl*(np.sqrt((1-sin2thetaW)/sin2thetaW) - 2*np.sqrt(sin2thetaW*(1-sin2thetaW)))

          m.dV_Z = 1 - 2*((sin2thetaW*(1-sin2thetaW))/(1-sin2thetaW-sin2thetaW))*(gha2_cpl-ghz2_cpl) - 2*np.sqrt(sin2thetaW*(1-sin2thetaW))*ghza2_cpl - (mZ**2)/(2*(1-sin2thetaW-sin2thetaW))*(ghz1prime2_cpl/lambda_Z1**2)
          m.dP_Z = 1 - sin2thetaW/(1-sin2thetaW-sin2thetaW)*(gha2_cpl-ghz2_cpl) - np.sqrt(sin2thetaW/(1-sin2thetaW))*ghza2_cpl - (mZ**2)/(2*(1-sin2thetaW-sin2thetaW))*(ghz1prime2_cpl/lambda_Z1**2)
          m.dM_Z = m.dP_Z
          m.dFour_Z = -np.sqrt(sin2thetaW/(1-sin2thetaW))*m.dFour_A

          m.dAAWpWm = 1
          m.dZAWpWm = m.dP_Z
          m.dZZWpWm = 2*m.dP_Z - 1
        
        if (prob != SampleHypothesisMCFM) and ("MCFM" in prob):
            #setting the Higgs mass and width for reweighting
#            m.setMelaHiggsMassWidth(125.38,0.004029,0)
            m.setMelaHiggsMassWidth(125,0.004029,0) 
            #print("Higgs mass is 125.38")
            #print("Higgs width is 0.004029")
        
        if couplings != [None]:
          # print('couplings!', couplings)
          for coupling_duo in couplings: #sets all the couplings set earlier
            setattr(m, coupling_duo[0], coupling_duo[1])
            # print(coupling_duo[0], "set to", getattr(m, coupling_duo[0]))
          
        if parsed_prob_dict["Prod"] == True and parsed_prob_dict["Dec"] == False:
          probdict[prob][0] = m.computeProdP()
        elif parsed_prob_dict["Prod"] == False and parsed_prob_dict["Dec"] == True:
          probdict[prob][0] = m.computeP()
        elif parsed_prob_dict["Prod"] == True and parsed_prob_dict["Dec"] == True:
          probdict[prob][0] = m.computeProdDecP()
        else:
          raise ValueError("Failed to process the probability passed here")
      if HasMCFMSampleHypothesis:
        for prob in probdict:
          if (prob != SampleHypothesisMCFM) and ("MCFM" in prob):
            probdict[prob][0] /= probdict[SampleHypothesisMCFM][0]
            # print(prob)
            # print()
        # Now divide the Sample Hypothesis by itself to make the probability = 1
        probdict[SampleHypothesisMCFM][0] /= probdict[SampleHypothesisMCFM][0]
      if HasJHUGenSampleHypothesis:
        for prob in probdict:
          if (prob != SampleHypothesisJHUGen) and ("JHUGen" in prob):
            
            if probdict[prob][0] == 0 or probdict[SampleHypothesisJHUGen][0] == 0:
              print('\ndivision with a zero!',probdict[prob][0],probdict[SampleHypothesisJHUGen][0])
              
            probdict[prob][0] /= probdict[SampleHypothesisJHUGen][0]
            #print(probdict[prob][0])
        # Now divide the Sample Hypothesis by itself to make the probability = 1
        probdict[SampleHypothesisJHUGen][0] /= probdict[SampleHypothesisJHUGen][0]
      #once at the end of the event
      m.resetInputEvent()
      newt.Fill()

      if i % 1000 == 0 or i == t.GetEntries():
        print(i, "/", t.GetEntries())
        
    newf.Write()
  except:
    os.remove(outfile)
    raise
