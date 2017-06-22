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

import os
import sys
if sys.version_info[0] == 3:
    from importlib import reload
elif sys.version_info[0] == 2:
    pass

import time
# import itertools 
import numpy as np
import sympy as sym
import networkx as nx
import matplotlib.pyplot as plt

import arrangement.geometricTraits as trts
reload(trts)
import arrangement.arrangement as arr
reload(arr)
import arrangement.plotting as aplt
reload(aplt)
import arrangement.utils as utls
reload(utls)


if not( os.path.abspath('./../dev/') in sys.path):
    sys.path.append( os.path.abspath('./../dev/') )
    
import my_svg_parser_dev as mSGVp
reload(mSGVp)

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

# loading a test case
test_case_id = 23
file_name = 'tests/testCases/'+test_cases_key[test_case_id]+'.yaml'

file_name = 'tests/testCases/curved_checkerboard_square.yaml'
file_name = 'tests/testCases/test.yaml'

# laoding a yaml file directly
# file_name = 'testCases/intel-01-occ-05cmintel-01-occ-05cm.yaml'

# loading yaml from svg
# file_name = 'testCases/svg_files/rect_circ_poly_line.svg' # fialed - result
# file_name = 'testCases/svg_files/rect_circ_poly_line_rotated.svg' # fialed - result
# file_name = 'testCases/svg_files/circle_lines.svg' # fialed - result
# file_name = 'testCases/svg_files/intel-01-occ-05cm.svg' # passed
# file_name = 'testCases/svg_files/hfab_110-001-06-0501.svg' # fialed - result
# file_name = 'testCases/svg_files/hfab_110-001-06-0501_rotated.svg' # xxx
# file_name = 'testCases/svg_files/hfab_110-001-06-0501_simplified.svg' # fialed - running
# file_name = 'testCases/svg_files/hfab_110-001-06-0501_simplified_rotated.svg' # fialed - running
# file_name = 'testCases/svg_files/islab_01.svg' # passed
# file_name = 'testCases/svg_files/islab_01_rotated.svg' # passed
# file_name = 'testCases/svg_files/islab_02.svg' # passed
# file_name = 'testCases/svg_files/islab_02_rotated.svg' # passed
# file_name = '/home/saesha/Dropbox/myGits/sample_data/arrangement_test_svg_files/circle_lines.svg'

if file_name.split('.')[-1] == 'svg':
    file_name = mSGVp.svg_to_ymal(file_name)#, convert_segment_2_infinite_line=True)

data = utls.load_data_from_yaml( file_name )
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
timing = False
visualize = True

print( '\nstart decomposition:', file_name)

tic = time.time()
config = {'multi_processing':4, 'end_point':False, 'timing':timing}
arrange = arr.Arrangement(traits, config)
if timing: print( 'Arrangement time:', time.time() - tic)

################################################################################
############################################# testing: if test data is available
################################################################################
if testing:
    cond =[]
    cond += [ len(arrange.graph.nodes()) == n_nodes ]
    cond += [ len(arrange.graph.edges()) == n_edges ]
    cond += [ len(arrange.decomposition.faces) == n_faces ]
    cond += [ len(arrange._subDecompositions) == n_subGraphs ]
    print( 'pass' if all(cond) else 'fail' )

    print( 'nodes:\t\t', len(arrange.graph.nodes()), '\t expected:', n_nodes )
    print( 'edges:\t\t', len(arrange.graph.edges()), '\t expected:', n_edges )
    print( 'faces:\t\t', len(arrange.decomposition.faces), '\t expected:', n_faces )
    print( 'subGraphs:\t', len(arrange._subDecompositions), '\t expected:', n_subGraphs )

elif not(testing):
    print( 'nodes:\t\t', len(arrange.graph.nodes()) )
    print( 'edges:\t\t', len(arrange.graph.edges()) )
    print( 'faces:\t\t', len(arrange.decomposition.faces) )
    print( 'subGraphs:\t', len(arrange._subDecompositions) )


################################################################################
################################################################## visualization
################################################################################
if visualize:

    # # static plotting
    # aplt.plot_graph(arrange.graph)
    # aplt.plot_decomposition_colored(arrange,
    #                                  printNodeLabels=False,
    #                                  printEdgeLabels=False)

    # aplt.plot_decomposition(arrange,
    #                          invert_axis = ('y'),
    #                          interactive_onClick=False,
    #                          interactive_onMove=False,
    #                          plotNodes=True, printNodeLabels=True,
    #                          plotEdges=True, printEdgeLabels=True)
    
    ############################## animated plotting
    # aplt.animate_halfEdges(arrange, timeInterval = 1.*1000)
    aplt.animate_face_patches(arrange, timeInterval = .5* 1000)

################################################################################
################################################################### testing area
################################################################################

# prime = arrange.get_prime_graph()
# dual  = arrange.get_dual_graph()


######################## plotting graphs
# myplt.plot_graph_pydot(arrange.graph)
# myplt.plot_graph_pydot(arrange.get_adjacency_graph())
# myplt.plot_graph_pydot(arrange.get_connectivity_graph())
# aplt.plot_multiple_graphs_pydot( [ arrange.graph ,
#                                    arrange.get_prime_graph(),
#                                    arrange.get_dual_graph() ] )








# plot_traits_from_file = True
# if plot_traits_from_file:
    
#     fig = plt.figure( figsize=(12, 12) )
#     ax = fig.add_subplot(111)
#     for t in traits: #arrange.curves:
#         ax.plot( [t.obj.p1.x, t.obj.p2.x], [t.obj.p1.y, t.obj.p2.y], 'b'  )
    
#     # for nodeIdx in arrange.graph.nodes():
#     #     p = arrange.graph.node[nodeIdx]['obj'].point
#     #     ax.plot( p.x, p.y, 'r*'  )

#     # ips = arrange.all_intersection_points
#     # for r in range(np.shape(ips)[0]):
#     #     for c in range(np.shape(ips)[1]):
#     #         for p in ips[r][c]:
#     #             if isinstance(p, sym.Point):
#     #                 ax.plot( p.x, p.y, 'r*')


#     for nodeIdx in arrange.graph.nodes():
#         p = arrange.graph.node[nodeIdx]['obj'].point
#         ax.plot( p.x, p.y, 'g^'  )
                    
#     # ax.plot( [488.65163,88.820421] , [4.6972099 ,31.62276] , 'r-.'  )
#     plt.axis('equal')
#     plt.tight_layout()
#     ax.invert_yaxis()    
#     plt.show()






################################################################################
########################################################################### Dump
################################################################################
''' switching from sympy to svg

>>> advantages:
1) higher speed for intersection procedure
intersection is about 97% of arrangement computation time
svgpathtools improves the intersection computation time 40 to 200 fold (depending on the trait type)

2) it has a much richer set for trait representation



>>> drawbacks
0) svgpathtools seems to be the only package to support geometric manipulation
specifically the intersection method!
(although pyx does it too, but it is postScript, and haven't tried it, so don't know much about the speed)

1) IDK how to get the univariate representation of the traits
which is the core requirement for my arrangement algorithms
so!

2) requires dev time

3) if use svgpathtools, it only offers Line, Arc, CubicBezier and QuadraticBezier
this means to give-up rays and infinit lines, and represent circles with arcs


>>> conclusion
I will use only the intersection procedure for only line segments, arcs and circles
line segments -> svgpathtools.Line
arcs -> svgpathtools.Arc
circles -> svgpathtools.Arc


'''




##############################
# the connected_component_subgraphs are not references to the other original graph, but to 
# updating superFace of _subDecompositions[2], does not update arrange.decomposition.faces[3]
# if subgraphs of sebdecompositions are to be relevant, every update applied to self.graph must be applied to self.subDecompositions.graph too. Or, store halfEdges separately and refer to them in the graph structure.
##############################

########################################
# #########SVG intersection is much much faster than sympy
########################################
# N = 10000
# ('sympy lxl time:\t', 10.229342937469482)
# ('svgpathtools lxl time:\t', 0.05286216735839844)
# ('sympy lxc time:\t', 182.2626440525055)
# ('svgpathtools lxa time:\t', 5.15391993522644)

# tic = time.time()
# sym_i = [ sym.intersection( sym.Line((0,0), (1,1)),
#                             sym.Line((0,0), (1,2)) )
#           for i in range(N) ]
# print( 'sympy lxl time:\t', time.time() - tic )

# import svgpathtools
# tic = time.time()
# svg_i = [ svgpathtools.Line(0+0j, 1+1j).intersect( svgpathtools.Line(0+0j, 1+2j) )
#           for i in range(N) ]
# print( 'svgpathtools lxl time:\t', time.time() - tic )

# tic = time.time()
# sym_i = [ sym.intersection( sym.Line((0,0), (1,1)),
#                             sym.Circle((1,0), 1) )
#           for i in range(N) ]
# print( 'sympy lxc time:\t', time.time() - tic )

# import svgpathtools
# tic = time.time()
# svg_i = [ svgpathtools.Line(0+0j, 1+1j).intersect( svgpathtools.Arc(start=-1, radius=1+1j, rotation=0, large_arc=1, sweep=1, end=2) )
#           for i in range(N) ]
# print( 'svgpathtools lxa time:\t', time.time() - tic )
########################################

# N = 10000
# x = np.arange(N)
# y = np.arange(N)

# tic = time.time()
# res_np = np.sqrt( np.diff(x)**2 + np.diff(y)**2 )
# print( 'np time:\t', time.time() - tic )

# tic = time.time()
# res_np_for = np.array([ np.sqrt( (x[idx]-x[idx+1])**2 + (y[idx]-y[idx+1])**2 )
#                         for idx in range(len(x)-1) ])
# print( 'np-for time:\t', time.time() - tic )


# tic = time.time()
# res_sym = np.array([ sym.sqrt( (x[idx]-x[idx+1])**2 + (y[idx]-y[idx+1])**2 )
#                      for idx in range(len(x)-1) ])
# print( 'np-for time:\t', time.time() - tic )


# print any(res_np_for-res_sym)

# print any(res_np_for-res_np)


###### 
# ips = np.array([[1,4],
#                 [2,5],
#                 [3,6]])

# xh = np.repeat( [ips[:,0]], ips.shape[0], axis=0)
# xv = np.repeat( [ips[:,0]], ips.shape[0], axis=0).T
# dx = xh - xv

# yh = np.repeat( [ips[:,1]], ips.shape[0], axis=0)
# yv = np.repeat( [ips[:,1]], ips.shape[0], axis=0).T
# dy = yh - yv

# dis = np.sqrt( dx**2 + dx**2)
