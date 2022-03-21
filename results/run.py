import os
import argparse
from tox21_models import interface 
from tox21_models import interface_binary
from tox21_models import interface_binary_deeptox
from tox21_models import interface_binary_multi

parser = argparse.ArgumentParser()
parser.add_argument("--save", help = "Save folder name")
parser.add_argument("-c", "--classification", help = "classification model",
        action='store_true')
parser.add_argument("-l", "--load", help = "load model",
        action='store_true')
parser.add_argument("-m", "--manual", help = "manual code",
        action='store_true')
parser.add_argument("-dim3", "--dim3", help = "dimnesions",
        action='store_true')
parser.add_argument("-DT", "--deeptox", help = "deeptox",
        action='store_true')
parser.add_argument("-mt", "--multi", help = "mutlitask",
        action='store_true')
args = parser.parse_args()

if args.dim3:
    if args.manual:
        name = 'Liver_Classification_Run'
        interface.run(['python '], name, False)
    #elif os.path.exists(args.save) and args.load is None:
        #print("This folder already exists, choose another one")
    else:
        if args.classification:
            interface.run(['Liver'], args.save, False)
        else:
            interface.run(['Liver'], args.save, True)

elif args.deeptox:
    print("DT in run is sex")
    if args.manual:
        name = 'Liver_Classification_Run'
        interface_binary_deeptox.run(['python '], name, False)
    #elif os.path.exists(args.save) and args.load is None:
        #print("This folder already exists, choose another one")
    else:
        if args.classification:
            interface_binary_deeptox.run(['Liver','Kidney', 'Chicken','Hamster','Apoptosis'], args.save, False)
        else:
            interface_binary_deeptox.run(['Liver','Kidney', 'Chicken','Hamster','Apoptosis'], args.save, True)

elif args.multi:
    print("multi run is sex")
    if args.manual:
        name = 'Liver_Classification_Run'
        interface_binary_multi.run(['python '], name, False)
    #elif os.path.exists(args.save) and args.load is None:
        #print("This folder already exists, choose another one")
    else:
        if args.classification:
            interface_binary_multi.run(['Liver'], args.save, False)
        else:
            interface_binary_multi.run(['Liver'], args.save, True)

else:
    if args.manual:
        name = 'Liver_Classification_Run'
        interface_binary.run(['python'], name, False)
    #elif os.path.exists(args.save) and args.load is None:
        #print("This folder already exists, choose another one")
    else:
        if args.classification:
            interface_binary.run(['Liver'], args.save, False)
        else:
            interface_binary.run(['Liver'], args.save, True)