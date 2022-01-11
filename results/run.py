import os
import argparse
from tox21_models import interface 

parser = argparse.ArgumentParser()
parser.add_argument("--save", help = "Save folder name")
parser.add_argument("-c", "--classification", help = "classification model",
        action='store_true')
parser.add_argument("-l", "--load", help = "load model",
        action='store_true')
parser.add_argument("-m", "--manual", help = "manual code",
        action='store_true')
args = parser.parse_args()


if args.manual:
    name = 'Liver_Classification_Run'
    interface.run(['Liver'], name, False)
#elif os.path.exists(args.save) and args.load is None:
    #print("This folder already exists, choose another one")
else:
    if args.classification:
        interface.run(['Liver'], args.save, False)
    else:
        interface.run(['Liver'], args.save, True)