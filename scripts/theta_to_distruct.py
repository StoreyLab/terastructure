import argparse

parser = argparse.ArgumentParser(description='Convert theta.txt to distruct compatible')
parser.add_argument("-o", "--output", dest="outfile",  help="path to outfile", required=True)
parser.add_argument("-i", "--infile", dest="infile", help="path to theta.txt", required=True)

args = parser.parse_args()
_in = file(args.infile, "r")
_out = file(args.outfile, "w")


for line in _in:
  chunks = line.split()
  id = chunks[0]
  theta = chunks[2:-1]
  out = ['0', id, '0', '0', ':'] + theta
  _out.write(" ".join(out) + "\n")
