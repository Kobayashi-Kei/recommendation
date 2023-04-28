import argparse

def arg_parse_from_commandline(argNameList):
    parser = argparse.ArgumentParser()
    for argName in argNameList:
        parser.add_argument(argName, help=argName)
    args = parser.parse_args()
    return args