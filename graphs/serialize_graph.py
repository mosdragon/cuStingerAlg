# Serializes karate.graph to use for graphviz

import sys

if len(sys.argv) != 2:
    print "Usage: python %s <graph>"
    exit(0)

graph = sys.argv[1]
f = open(graph, 'r')

lines = f.read().splitlines()

NV = int(lines[0].split(' ', 1)[0])

for i in range(1, NV + 1):
    s = lines[i]
    print str(i) + " -- { " + s + " }; "
    
