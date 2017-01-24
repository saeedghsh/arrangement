'''
Copyright (C) Saeed Gholami Shahbandi. All rights reserved.
Author: Saeed Gholami Shahbandi (saeed.gh.sh@gmail.com)

This file is part of Arrangement Library.
The of Arrangement Library is free software: you can redistribute it and/or
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

import arrangement as arr
import plotting as myplt
import modifiedSympy as mSym
reload(arr)
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
    'example_10',                   # 26: a square with a loose branch
    'example_11',                   # 27: 3 circle holes in one square
    'caisr',                        # 28: 
    'specialCase_11',               # 29: 
    'example_12',                   # 30: 3 circle holes in one square, + 1xArc

]

test_case_id = 15
timing = False
visualize = True

file_name = 'testCases/'+test_cases_key[test_case_id]+'.yaml'
file_name = 'intel-01-occ-05cmintel-01-occ-05cm.yaml'


#### loading SVG files
if 1:
    import my_svg_parser as mSVGp
    reload(mSVGp)
    svg_file_name = 'testCases/rect_circ_poly_line.svg'
    svg_file_name = 'testCases/circle_lines.svg'
    svg_file_name = 'testCases/intel-01-occ-05cm.svg'
    file_name = mSVGp.svg_to_ymal(svg_file_name)


data = load_data_from_yaml( file_name )
traits = data['traits']
# traits += [mSym.ArcModified( args=( (4,4), 3 , (np.pi/10, 19*np.pi/10 ) ) )]

if 'number_of_nodes' in data.keys():
    testing = True
    n_nodes = data['number_of_nodes']
    n_edges = data['number_of_edges']
    n_faces = data['number_of_faces']
    n_subGraphs = data['number_of_subGraphs']
else:
    testing = False

################################################################################
###################################### deploying arrangement (and decomposition)
################################################################################
print( '\nstart decomposition:', test_cases_key[test_case_id])

tic = time.time()
config = {'multi_processing':4, 'end_point':False}
arrang = arr.Arrangement(traits, config)
if timing: print( 'Arrangement time:', time.time() - tic)

################################################################################
############################################# testing: if test data is available
################################################################################
if testing:
    cond =[]
    cond += [ len(arrang.graph.nodes()) == n_nodes ]
    cond += [ len(arrang.graph.edges()) == n_edges ]
    cond += [ len(arrang.decomposition.faces) == n_faces ]
    cond += [ len(arrang._subDecompositions) == n_subGraphs ]
    print( 'pass' if all(cond) else 'fail' )

    print( 'nodes:\t\t', len(arrang.graph.nodes()), '\t expected:', n_nodes )
    print( 'edges:\t\t', len(arrang.graph.edges()), '\t expected:', n_edges )
    print( 'faces:\t\t', len(arrang.decomposition.faces), '\t expected:', n_faces )
    print( 'subGraphs:\t', len(arrang._subDecompositions), '\t expected:', n_subGraphs )

elif not(testing):
    print( 'nodes:\t\t', len(arrang.graph.nodes()) )
    print( 'edges:\t\t', len(arrang.graph.edges()) )
    print( 'faces:\t\t', len(arrang.decomposition.faces) )
    print( 'subGraphs:\t', len(arrang._subDecompositions) )


################################################################################
################################################################## visualization
################################################################################
if visualize:
    # # static plotting
    # # myplt.plot_graph(arrang.graph)
    # myplt.plot_decomposition_colored(arrang,
    #                                  printNodeLabels=False,
    #                                  printEdgeLabels=False)

    # myplt.plot_decomposition(arrang,
    #                          interactive_onClick=False,
    #                          interactive_onMove=False,
    #                          plotNodes=True, printNodeLabels=True,
    #                          plotEdges=True, printEdgeLabels=True)
    
    
    ############################## animated plotting
    # myplt.animate_halfEdges(arrang, timeInterval = 1.*1000)
    myplt.animate_face_patches(arrang, timeInterval = .5*1000)


    # ######################## plotting graphs
    # # myplt.plot_graph_pydot(arrang.graph)
    # # myplt.plot_graph_pydot(arrang.get_adjacency_graph())
    # # myplt.plot_graph_pydot(arrang.get_connectivity_graph())
    # myplt.plot_multiple_graphs_pydot( [ arrang.graph,
    #                                     arrang.get_adjacency_graph(),
    #                                     arrang.get_connectivity_graph() ] )


################################################################################
################################################################### testing area
################################################################################



################################################################################
########################################################################### Dump
################################################################################

##############################
# the connected_component_subgraphs are not references to the other original graph, but to 
# updating superFace of _subDecompositions[2], does not update arrang.decomposition.faces[3]
# if subgraphs of sebdecompositions are to be relevant, every update applied to self.graph must be applied to self.subDecompositions.graph too. Or, store halfEdges separately and refer to them in the graph structure.
##############################
