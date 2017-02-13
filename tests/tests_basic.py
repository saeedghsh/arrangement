'''
Copyright (C) Saeed Gholami Shahbandi. All rights reserved.
Author: Saeed Gholami Shahbandi (saeed.gh.sh@gmail.com)

This file is part of Arrangision Library.
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
import unittest

import numpy as np

if not( os.path.abspath('./../') in sys.path):
    sys.path.append( os.path.abspath('./../') )

import arrangement.arrangement as arr
from arrangement.utils import load_data_from_yaml

################################################################################
################################################################## testing class
################################################################################

class ArrangementTests(unittest.TestCase):    
    # '''  '''
    # def test_twinAssignment(self, arrang):
    #     # checking the correctness of twin assignment in half-edge construction
        
    #     res = []
    #     for selfIdx in arrang.get_all_HalfEdge_indices():
    #         (s,e,k) = selfIdx
            
    #         twinIdx = arrang.graph[s][e][k]['obj'].twinIdx
    #         (ts,te,tk) = twinIdx

    #         twinOfTwinIdx = arrang.graph[ts][te][tk]['obj'].twinIdx            

    #         res += [ selfIdx == twinOfTwinIdx ]

    #     return all(res)

    def runTest(self):

        fileList = [ fileName
                     for fileName in os.listdir(address)
                     if (len(fileName)>5 and fileName[-5:]=='.yaml') ]
        
        
        for fileIdx, fileName in enumerate(sorted(fileList)):
            print (fileName)
            data = load_data_from_yaml( address+fileName )
            print('testing case: ' + data['dataset'] , '-\t', fileIdx+1, '/', len(fileList))
            traits = data['traits']
            config = {'multi_processing':4, 'end_point':False}
            arrang = arr.Arrangement(traits, config)

            # ########## testing twin assignment
            # self.assertEqual( self.test_twinAssignment(arrang), True,
            #                   'incorrect twin assignment')

            ########## testing number of nodes
            if 'number_of_nodes' in data.keys():
                n_nodes = data['number_of_nodes']
                self.assertEqual( len(arrang.graph.nodes()), n_nodes, 'incorrect number of nodes')
            else:
                print('number of nodes is not available for ' + key, '...')

            ########## testing number of edges
            if 'number_of_edges' in data.keys():
                n_edges = data['number_of_edges']
                self.assertEqual( len(arrang.graph.edges()), n_edges,
                                  'incorrect number of edges')
            else:
                print( 'number of edges is not available for ' + key, '...' )

            ########## testing number of faces
            if 'number_of_faces' in data.keys():
                n_faces = data['number_of_faces']
                self.assertEqual( len(arrang.decomposition.faces), n_faces,
                                  'incorrect number of faces')
            else:
                print( 'number of faces is not available for ' + key, '...' )

            ########## testing number of subGraphs
            if 'number_of_subGraphs' in data.keys():
                n_subGraphs = data['number_of_subGraphs']
                self.assertEqual( len(arrang._subDecompositions), n_subGraphs,
                                  'incorrect number of subGraphs')
            else:
                print( 'number of subGraphs is not available for ' + key, '...' )


            # ########## testing neighbourhood function [incomplete]
            # ########## it checks if neighbourhood is valid in both direction
            # for fIdx in range(len(arrang.decomposition.faces)):
            #     for nfIdx in arrang.decomposition.find_neighbours(fIdx):
            #         assert fIdx in arrang.decomposition.find_neighbours(nfIdx)



            # ########## testing transformation [incomplete]
            # ########## it doesn't check the correctness
            # ########## it just checks whether if it completes the process
            # arrang.transform_sequence('SRT',
            #                           ((5,5), -np.pi/2, (-10,0), ),
            #                           ((0,0), (0,0),    (0,0), ) )
   


if __name__ == '__main__':
    global address
    address = os.getcwd()+'/testCases/'

    unittest.main()



# ########################################
# # test node construction
# # does myArrangement.nodes correspond to myArrangement.intersectionFlat?
# num_of_intersections = len( myArrangement.intersectionsFlat)
# num_of_nodes = len( myArrangement.nodes )
# asser (num_of_intersections == num_of_nodes)

# for (n, p) in zip (myArrangement.nodes, myArrangement.intersectionsFlat):
#     n_point = myArrangement.nodes[n_idx][1]['obj'].point
#     assert(n_point.compare(p) == 0)

# # are nodes assigned correctly to traits?
# for c_idx, trait in enumerate(myArrangement.traits):
#     for n_idx, node in enumerate(myArrangement.nodes):
#         point = myArrangement.nodes[n_idx][1]['obj'].point
#         if trait.obj.contains(point):
#             if not( c_idx in myArrangement.ipsTraitIdx[n_idx] ):
#                 print( 'error' )
# ########################################


# ########################################
# # half edge attributes:
# TODO:
# assert (twinIdx[1] == selfIdx[0])
# assert (twinIdx[1] == selfIdx[0])
# assert (succIdx[0] == selfIdx[1])
# ########################################        


############################# test halfEdge tvals
# nodes = [ myArrangement.graph.node[key]['obj'] for key in myArrangement.graph.node.keys()]
# for s,e,k in myArrangement.get_all_HalfEdge_indices():
#     he = myArrangement.graph[s][e][k]['obj']

#     sTVal = nodes[s].traitTval[nodes[s].traitIdx.index(he.cIdx)]
#     eTVal = nodes[e].traitTval[nodes[e].traitIdx.index(he.cIdx)]
    
#     if (he.direction=='positive') and not(sTVal < eTVal):
#         eTVal += 2*np.pi
#     if (he.direction=='negative') and not(sTVal > eTVal):
#         sTVal += 2*np.pi

#     assert he.sTVal == sTVal
#     assert he.eTVal == eTVal
# ########################################

# ########################################
# # test face construction
# # could a pair of twins be in oen face simultaneously? YES! They could!

# allHalfEdge = myArrangement.get_all_HalfEdge_indices()
# he = (2,8,0)
# idx = myArrangement.find_successor_HalfEdge(he)
# he = allHalfEdge[idx]
# print (he)
# myArrangement.find_successor_HalfEdge(he)
# ########################################

# ########################################
# # testing the successor/twin assigment
# for face in myArrangement.decomposition.faces:
#     for idx in range(len(face.halfEdges)-1):
#         (cs,ce,ck) = face.halfEdges[idx] # current halfEdgeIdx
#         (ss,se,sk) = face.halfEdges[idx+1] # successor halfEdgeIdx
#         assert (myArrangement.graph[cs][ce][ck]['obj'].succIdx == (ss,se,sk))
        
#         (ts,te,tk) = myArrangement.MDG[cs][ce][ck]['obj'].twinIdx
#         assert (ts == ss)

#     (cs,ce,ck) = face.halfEdges[-1] # current halfEdgeIdx
#     (ss,se,sk) = face.halfEdges[0] # successor halfEdgeIdx
#     assert (myArrangement.graph[cs][ce][ck]['obj'].succIdx == (ss,se,sk))

#     (ts,te,tk) = myArrangement.graph[cs][ce][ck]['obj'].twinIdx
#     assert (ts == ss)
# ########################################

