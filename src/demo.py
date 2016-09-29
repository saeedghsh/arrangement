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

import time
import numpy as np
import sympy as sym

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
]

########################################
testNumber = 27
timing = False

file_name = 'testCases/'+test_cases_key[testNumber]+'.yaml'
data = load_data_from_yaml( file_name )
curves = data['curves']

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
print '\nstart decomposition:', test_cases_key[testNumber]

tic = time.time()
mySubdivision = sdv.Subdivision(curves, multiProcessing=4)
if timing: print 'Subdivision time:', time.time() - tic

if testing:
    cond =[]
    cond += [ len(mySubdivision.MDG.nodes()) == n_nodes ]
    cond += [ len(mySubdivision.MDG.edges()) == n_edges ]
    cond += [ len(mySubdivision.decomposition.faces) == n_faces ]
    cond += [ len(mySubdivision.subDecompositions) == n_subGraphs ]
    print 'success' if all(cond) else 'fail'

    print 'nodes:\t\t', len(mySubdivision.MDG.nodes()), '\t expected:', n_nodes
    print 'edges:\t\t', len(mySubdivision.MDG.edges()), '\t expected:', n_edges
    print 'faces:\t\t', len(mySubdivision.decomposition.faces), '\t expected:', n_faces
    print 'subGraphs:\t', len(mySubdivision.subDecompositions), '\t expected:', n_subGraphs

elif not(testing):
    print 'nodes:\t\t', len(mySubdivision.MDG.nodes())
    print 'edges:\t\t', len(mySubdivision.MDG.edges())
    print 'faces:\t\t', len(mySubdivision.decomposition.faces)
    print 'subGraphs:\t', len(mySubdivision.subDecompositions)

################################################################################
####################################################################### plotting
################################################################################
# myplt.plot_graph(mySubdivision.MDG)
# myplt.plot_decomposition_colored(mySubdivision,
#                                  printNodeLabels=False,
#                                  printEdgeLabels=False)

# myplt.plot_decomposition(mySubdivision,
#                          interactive_onClick=False,
#                          interactive_onMove=False,
#                          plotNodes=True, printNodeLabels=True,
#                          plotEdges=True, printEdgeLabels=True)

################################################################################
###################################################################### animating
################################################################################
# myplt.animate_halfEdges(mySubdivision, timeInterval = 1.*1000)
# myplt.animate_face_patches(mySubdivision, timeInterval = .5*1000)

################################################################################
####################################################################### API demo
################################################################################
'''
important note about nodes of networkx:
MDG.nodes() is not neccessarily [0,1,2,...]
it's important to remember that MDG.node is a dict, not a list
MDG.node[idx] is not actually indexing the MDG.node, but fetching from a dict
'''

# subdiv = mySubdivision
# subdiv.subGraphs, subdiv.subDecompositions
# subdiv.MDG, subdiv.decomposition

# # access nodes of networkX graph and their attribute dict
# for nodeIdx in subdiv.MDG.nodes():
#     print nodeIdx, ': ', subdiv.MDG.node[nodeIdx]

# # access edges of networkX graph:
# for halfEdgeIdx in subdiv.get_all_HalfEdge_indices():
#     (s,e,k) = (startNodeIdx, endNodeIdx, path) = halfEdgeIdx
#     print (s,e,k), ': ', subdiv.MDG[s][e][k]['obj']

# # access main graph of subdivision:




############################# find cycles?
# import networkx as ns
# MDG = mySubdivision.MDG
# MG = MDG.copy()
# allHalfEdgeIdx = [(sIdx, eIdx, k)
#                   for sIdx in MDG.nodes()
#                   for eIdx in MDG.nodes()
#                   if eIdx in MDG[sIdx].keys() # if not, subd[sIdx][eIdx] is invalid
#                   for k in MDG[sIdx][eIdx].keys()]
# twins = []
# for hei in allHalfEdgeIdx:
#     (s,e,k) = hei
#     twin = MG[s][e][k]['obj'].twinIdx
#     if not( hei in twins) :
#         twins.append(twin)
# MG.remove_edges_from(twins)
# MG = MG.to_undirected()
# for cycle in nx.cycles.simple_cycles(MG):
#     print cycle
# for cycle in nx.cycles.cycle_basis(MG):
#     print cycle



# ### testing the successor/twin assigment
# for face in mySubdivision.decomposition.faces:
#     for idx in range(len(face.halfEdges)-1):
#         (cs,ce,ck) = face.halfEdges[idx] # current halfEdgeIdx
#         (ss,se,sk) = face.halfEdges[idx+1] # successor halfEdgeIdx
#         assert (mySubdivision.MDG[cs][ce][ck]['obj'].succIdx == (ss,se,sk))
        
#         (ts,te,tk) = mySubdivision.MDG[cs][ce][ck]['obj'].twinIdx
#         assert (ts == ss)

#     (cs,ce,ck) = face.halfEdges[-1] # current halfEdgeIdx
#     (ss,se,sk) = face.halfEdges[0] # successor halfEdgeIdx
#     assert (mySubdivision.MDG[cs][ce][ck]['obj'].succIdx == (ss,se,sk))

#     (ts,te,tk) = mySubdivision.MDG[cs][ce][ck]['obj'].twinIdx
#     assert (ts == ss)

