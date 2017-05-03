import networkx as nx
import fileinput

# params
K = 256

edges = {}
G = nx.Graph()

for num, line in enumerate(fileinput.input()):
    if num == 0:
        nv, ne, _ = [int(x) for x in line.split()]
    else:
        src = num - 1  # src
        line = line.strip()
        vals = [int(x) for x in line.split()]
        for dst in vals:
            G.add_edge(src, dst)

bc_vals = nx.betweenness_centrality(G)
for i in sorted(bc_vals.keys()):
    print "{}: {}".format(i, bc_vals[i])
    if bc_vals[i] > 1:
        print "OMG: {}-{}".format(i, bc_vals[i])
        
