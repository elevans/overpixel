import argparse
from . import overlap
from . import util

def parse_args():
    description = "Pixel correlating tools."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input', '-i', required=True, type=str, help="Path to input .tif file.")
    parser.add_argument('--method', '-m', choices=['binary_overlap', 'saca'], required=True, help="Method to analyze the dataset.")
    parser.add_argument('--show', '-s', action='store_true', help='Show results.')
    parser.add_argument('--batch', '-b', action='store_true', help='Set the input as a batch job.')

    return parser.parse_args()

def main():
    args = parse_args()
    if args.method == 'binary_overlap':
        results = overlap.compute(util.load_file(args.input), show=args.show)
        print(f"[DEBUG]: {results}")

if __name__ == "__main__":
    main()