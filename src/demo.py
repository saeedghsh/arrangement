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

import subdivision as sdv
import plotting as myplt
import testCaseGenerator as tcg
reload(sdv)
reload(myplt)
reload(tcg)

# TODO(saesha) documentation

# TODO(saesha) testing 
# use suite() so it runs all tests, even if one fails

# TODO(saesha) bug "test case 2" fails
# two tangent circles one inside another are not standalone, but behave like one
# starting from the positive half-edge of the outer circle, the face will close
# with only one half-edge, ignoring the negative half-edge of the inner circle.
# that is because the condition of sNodeIdx == eNodeIdx will be satisfied.

# TODO(saesha) bug "test case 13" fails
# due to unknown

# TODO(saesha) bug-developement
# for hole in face.holes: assert len(hole.face.hole)==0

# TODO(saesha) developement
# Decomposition.find_neighbours(), and don't forget to include half-edges from holes.
# for this actually we need to make sure there is no redundancy in holes!


################################################################################
################################################################# initialization
################################################################################
test_cases_key = [
    'Default',                      # 0: 
    'Default_Small',                # 1: 
    'special_case_01',              # 2: 4 tangent circles and a line
    'special_case_02',              # 3: non-interrsecting circle
    'special_case_03',              # 4: singel circle 4 intersections
    'special_case_04',              # 5: 2 lines - one circle
    'special_case_05',              # 6: a square - a circle
    'special_case_06',              # 7: concentric circles
    'special_case_07',              # 8: disjoint
    'intensive',                    # 9: 
    'intensive_x2',                 # 10:
    'intensive_only_line',          # 11:
    'Random',                       # 12:
    'star',                         # 13: 
    'example1'                      # 14: 2 circles and a line
]

test_case = tcg.get_test(name=test_cases_key[14])
curves = test_case['curves']
n_nodes = test_case['number_of_nodes']
n_edges = test_case['number_of_edges']
n_faces = test_case['number_of_faces']
n_subGraphs = test_case['number_of_subGraphs']

################################################################################
################################################## subdivision and decomposition
################################################################################
print 'start decomposition'
mySubdivision = sdv.Subdivision(curves, multiProcessing=4)
print 'nodes:\t\t', len(mySubdivision.MDG.nodes()), '\t expected:', n_nodes
print 'edges:\t\t', len(mySubdivision.MDG.edges()), '\t expected:', n_edges
print 'faces:\t\t', len(mySubdivision.decomposition.faces), '\t expected:', n_faces
print 'subGraphs:\t', len(mySubdivision.subGraphs), '\t expected:', n_subGraphs

################################################################################
####################################################################### plotting
################################################################################
# myplt.plot_graph(mySubdivision.MDG)
# myplt.plot_decomposition_colored(mySubdivision,
#                                  printNodeLabels=False,
#                                  printEdgeLabels=False)
myplt.plot_decomposition(mySubdivision,
                         interactive_onClick=False,
                         interactive_onMove=False,
                         plotNodes=False, printNodeLabels=False,
                         plotEdges=True, printEdgeLabels=False)

################################################################################
###################################################################### animating
################################################################################
myplt.animate_face_patches(mySubdivision)
# myplt.animate_halfEdges(mySubdivision)


################################################################################
####################################################################### API demo
################################################################################
# Subdivision
# Decomposition
# Curve
# Node
# HalfEdge
# Face
