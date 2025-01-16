import os
import argparse
from ROOT import TFile, TTree, TList, TChain, RDataFrame

def check_existence(filepath):
    if not os.path.exists(filepath):
        raise argparse.ArgumentTypeError(f"File {filepath} not found!")
    
    return os.path.abspath(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", nargs=2, required=True, type=check_existence)
    parser.add_argument("-t", "--tTree", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)

    pathList = args.files

    f1 = TFile(pathList[0], 'read')
    t1 = f1.Get(args.tTree)

    f2 = TFile(pathList[1], 'read')
    t2 = f2.Get(args.tTree)

    t1.AddFriend(t2)

    df = RDataFrame(t1)

    print("Creating output file")

    df.Snapshot(args.tTree, args.output)
