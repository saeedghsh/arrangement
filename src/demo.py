'''
Copyright (C) Saeed Gholami Shahbandi. All rights reserved.
Author: Saeed Gholami Shahbandi (saeed.gh.sh@gmail.com)

This file is part of Subdivision Library.
The of Subdivision Library is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this program. If not, see <http://www.gnu.org/licenses/>
'''

from __future__ import print_function

import sys
if sys.version_info[0] == 3:
    from importlib import reload
elif sys.version_info[0] == 2:
    pass

import time
import itertools 
import numpy as np
import sympy as sym
import networkx as nx
import matplotlib.pyplot as plt

import subdivision as sdv
import plotting as myplt
import modifiedSympy as mSym
reload(sdv)
reload(myplt)
reload(mSym)
from loadFromYaml import load_data_from_yaml

################################################################################
################################################################# initialization
################################################################################
test_cases_key = [
    'default',                      # 0: 
    'default_small',                # 1: 
    'specialCase_01',               # 2: 4 tangent circles and a line
    'specialCase_02',               # 3: non-interrsecting circle
    'specialCase_03',               # 4: single circle 4 intersections
    'specialCase_04',               # 5: 2 lines - one non-interrsecting circle
    'specialCase_05',               # 6: a square - a circle - with no interrsecting
    'specialCase_06',               # 7: concentric circles
    'specialCase_07',               # 8: disjoint
    'specialCase_08',               # 9: 2 pairs of tangent circles
    'intensive',                    # 10: 
    'intensive_x2',                 # 11:
    'intensive_lineOnly',           # 12:
    'random',                       # 13:
    'star',                         # 14: 
    'example_01',                   # 15: 2 circles and a line
    'specialCase_09',               # 16: 
    'specialCase_10',               # 17: 
    'example_02',                   # 18: +arc +segment +ray
    'example_03',                   # 19: only segment
    'example_04',                   # 20: pizza: circle + 8 rays
    'example_05',                   # 21: capsule: 2xArcs + 2xSements
    'example_06',                   # 22: vertically stacked arcs and rays
    'example_07',                   # 23: door
    'example_08',                   # 24: 
    'example_09',                   # 25: 3x segment-> 2xNode-1xEdge-0xFace
    'example_10',                   # 26: a square with a loost branch
    'example_11',                   # 27: 3 circle holes in one square
    'caisr',                        # 28: 
    'specialCase_11',               # 29: 
    'example_12',                   # 30: 3 circle holes in one square, + 1xArc

]

test_case_id = 8
timing = False
visualize = True

file_name = 'testCases/'+test_cases_key[test_case_id]+'.yaml'
data = load_data_from_yaml( file_name )
curves = data['curves']
# curves += [mSym.ArcModified( args=( (4,4), 3 , (np.pi/10, 19*np.pi/10 ) ) )]

if 'number_of_nodes' in data.keys():
    testing = True
    n_nodes = data['number_of_nodes']
    n_edges = data['number_of_edges']
    n_faces = data['number_of_faces']
    n_subGraphs = data['number_of_subGraphs']
else:
    testing = False

################################################################################
###################################### deploying subdivision (and decomposition)
################################################################################
print( '\nstart decomposition:', test_cases_key[test_case_id])

tic = time.time()
config = {'multi_processing':4, 'end_point':False}
subdiv = sdv.Subdivision(curves, config)
if timing: print( 'Subdivision time:', time.time() - tic)

################################################################################
############################################# testing: if test data is available
################################################################################
if testing:
    cond =[]
    cond += [ len(subdiv.graph.nodes()) == n_nodes ]
    cond += [ len(subdiv.graph.edges()) == n_edges ]
    cond += [ len(subdiv.decomposition.faces) == n_faces ]
    cond += [ len(subdiv._subDecompositions) == n_subGraphs ]
    print( 'pass' if all(cond) else 'fail' )

    print( 'nodes:\t\t', len(subdiv.graph.nodes()), '\t expected:', n_nodes )
    print( 'edges:\t\t', len(subdiv.graph.edges()), '\t expected:', n_edges )
    print( 'faces:\t\t', len(subdiv.decomposition.faces), '\t expected:', n_faces )
    print( 'subGraphs:\t', len(subdiv._subDecompositions), '\t expected:', n_subGraphs )

elif not(testing):
    print( 'nodes:\t\t', len(subdiv.graph.nodes()) )
    print( 'edges:\t\t', len(subdiv.graph.edges()) )
    print( 'faces:\t\t', len(subdiv.decomposition.faces) )
    print( 'subGraphs:\t', len(subdiv._subDecompositions) )


################################################################################
################################################################## visualization
################################################################################
if visualize:
    ############################### plotting
    # myplt.plot_graph(subdiv.graph)
    # myplt.plot_decomposition_colored(subdiv,
    #                                  printNodeLabels=False,
    #                                  printEdgeLabels=False)
    myplt.plot_decomposition(subdiv,
                             interactive_onClick=False,
                             interactive_onMove=False,
                             plotNodes=True, printNodeLabels=True,
                             plotEdges=True, printEdgeLabels=True)
    
    
    ############################## animating
    # myplt.animate_halfEdges(subdiv, timeInterval = 1.*1000)
    myplt.animate_face_patches(subdiv, timeInterval = .5*1000)


################################################################################
################################################################### testing area
################################################################################

'''
>>> construct adjacency and connectivity:
subdiv.graph is the main graph. I contains all sub-graphs (disconnected).
subdiv.decomposition is the main decomposition. I contains all the faces.

To construct adjacency and connectivity, just use:
subdiv.graph -> adjacency
subdiv.decomposition -> connectivity

try to plot the conncetivity graphs on top of the subdivision, to do so, I could ignore the geometric incorrectness of the edges, but atleast I need geometric location of the nodes.

'''


TODO: plot using graphviz
# http://stackoverflow.com/questions/14943439/how-to-draw-multigraph-in-networkx-using-matplotlib-or-graphviz


for halfEdgeIdx in subdiv.graph.edges(keys=True):
    (s,e,k) = (startNodeIdx, endNodeIdx, path) = halfEdgeIdx
    print (s,e,k), ': ', subdiv.graph[s][e][k]['obj'].attributes

nx.draw_networkx(subdiv.graph)
plt.show()

print (len(subdiv.graph), len(subdiv.graph.edges(keys=True)))

adjacency = subdiv.graph.to_undirected()
for halfEdgeIdx in adjacency.edges(keys=True):
    (s,e,k) = (startNodeIdx, endNodeIdx, path) = halfEdgeIdx
    print (s,e,k), ': ', adjacency[s][e][k]['obj'].attributes

nx.draw_networkx(adjacency)
plt.show()

print (len(adjacency), len(adjacency.edges(keys=True)) )


connectivity = nx.MultiGraph()
nodes = [ [fIdx, {'face':face}] for fIdx,face in enumerate(subdiv.decomposition.faces)]
connectivity.add_nodes_from( nodes )

for (f1Idx,f2Idx) in itertools.combinations( range(len(subdiv.decomposition.faces) ), 2):
    mutualsHalfEdges = subdiv.decomposition.find_mutual_halfEdges(f1Idx, f2Idx)
    if len(mutualsHalfEdges) !=0 :
        connectivity.add_edges_from( [ (f1Idx,f2Idx, {'mutualsHalfEdges': mutualsHalfEdges}) ] )

nx.draw_networkx(connectivity)
plt.show()



    


################################################################################
########################################################################### Dump
################################################################################

##############################
# the connected_component_subgraphs are not references to the other original graph, but to 
# updating superFace of _subDecompositions[2], does not update subdiv.decomposition.faces[3]
# if subgraphs of sebdecompositions are to be relevant, every update applied to self.graph must be applied to self.subDecompositions.graph too. Or, store halfEdges separately and refer to them in the graph structure.
##############################
