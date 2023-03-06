import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random Seed (Default: 42)",
    )
    parser.add_argument(
        "--level",
        type=str,
        metavar="basic | defend | deadly",
        help="The Level For The Game",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Initiate Training Mode")
    parser.add_argument(
        "--curr",
        action="store_true",
        help="Initiate Curriculum Learning Mode",
    )
    mode.add_argument("--test", action="store_true", help="Initiate Testing Mode")
    parser.add_argument(
        "--skill",
        type=int,
        help="Specify Enemy Skill Level",
        required="--level" == "deadly",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Specify a model to be loaded based on time-step",
        required="--test" in sys.argv,
    )

    args = parser.parse_args()
    return args
