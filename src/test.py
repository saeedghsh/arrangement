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

import os
import unittest

import subdivision as sdv
reload(sdv)

from loadFromYaml import load_data_from_yaml

################################################################################
################################################################## testing class
################################################################################

class SubdivisionTests(unittest.TestCase):    
    # '''  '''
    # def test_twinAssignment(self, subdiv):
    #     # checking the correctness of twin assignment in half-edge construction
        
    #     res = []
    #     for selfIdx in subdiv.get_all_HalfEdge_indices():
    #         (s,e,k) = selfIdx
            
    #         twinIdx = subdiv.graph[s][e][k]['obj'].twinIdx
    #         (ts,te,tk) = twinIdx

    #         twinOfTwinIdx = subdiv.graph[ts][te][tk]['obj'].twinIdx            

    #         res += [ selfIdx == twinOfTwinIdx ]

    #     return all(res)

    def runTest(self):

        fileList = [ fileName
                     for fileName in os.listdir(address)
                     if (len(fileName)>5 and fileName[-5:]=='.yaml') ]

        for fileIdx, fileName in enumerate(sorted(fileList)):
            
            data = load_data_from_yaml( address+fileName )
            print 'testing case: ' + data['dataset'] , '-\t', fileIdx+1, '/', len(fileList)
            curves = data['curves']            
            subdiv = sdv.Subdivision(curves, multiProcessing=4)

            # ########## testing twin assignment
            # self.assertEqual( self.test_twinAssignment(subdiv), True,
            #                   'incorrect twin assignment')

            ########## testing number of nodes
            if 'number_of_nodes' in data.keys():
                n_nodes = data['number_of_nodes']
                self.assertEqual( len(subdiv.graph.nodes()), n_nodes, 'incorrect number of nodes')
            else:
                print 'number of nodes is not available for ' + key, '...'

            ########## testing number of edges
            if 'number_of_edges' in data.keys():
                n_edges = data['number_of_edges']
                self.assertEqual( len(subdiv.graph.edges()), n_edges,
                                  'incorrect number of edges')
            else:
                print 'number of edges is not available for ' + key, '...'

            ########## testing number of faces
            if 'number_of_faces' in data.keys():
                n_faces = data['number_of_faces']
                self.assertEqual( len(subdiv.decomposition.faces), n_faces,
                                  'incorrect number of faces')
            else:
                print 'number of faces is not available for ' + key, '...'

            ########## testing number of subGraphs
            if 'number_of_subGraphs' in data.keys():
                n_subGraphs = data['number_of_subGraphs']
                self.assertEqual( len(subdiv.subDecompositions), n_subGraphs,
                                  'incorrect number of subGraphs')
            else:
                print 'number of subGraphs is not available for ' + key, '...'
            


if __name__ == '__main__':
    global address
    address = os.getcwd()+'/testCases/'

    unittest.main()



# ########################################
# # test node construction
# # does mySubdivision.nodes correspond to mySubdivision.intersectionFlat?
# num_of_intersections = len( mySubdivision.intersectionsFlat)
# num_of_nodes = len( mySubdivision.nodes )
# asser (num_of_intersections == num_of_nodes)

# for (n, p) in zip (mySubdivision.nodes, mySubdivision.intersectionsFlat):
#     n_point = mySubdivision.nodes[n_idx][1]['obj'].point
#     assert(n_point.compare(p) == 0)

# # are nodes assigned correctly to curves?
# for c_idx, curve in enumerate(mySubdivision.curves):
#     for n_idx, node in enumerate(mySubdivision.nodes):
#         point = mySubdivision.nodes[n_idx][1]['obj'].point
#         if curve.obj.contains(point):
#             if not( c_idx in mySubdivision.ipsCurveIdx[n_idx] ):
#                 print 'error'
# ########################################


# ########################################
# # half edge attributes:
# TODO:
# assert (twinIdx[1] == selfIdx[0])
# assert (twinIdx[1] == selfIdx[0])
# assert (succIdx[0] == selfIdx[1])
# ########################################        



# ########################################
# # test face construction
# # could a pair of twins be in oen face simultaneously? YES! They could!

# allHalfEdge = mySubdivision.get_all_HalfEdge_indices()
# he = (2,8,0)
# idx = mySubdivision.find_successor_HalfEdge(he)
# he = allHalfEdge[idx]
# print he
# mySubdivision.find_successor_HalfEdge(he)
# ########################################





# ########################################
# # testing the successor/twin assigment
# for face in mySubdivision.decomposition.faces:
#     for idx in range(len(face.halfEdges)-1):
#         (cs,ce,ck) = face.halfEdges[idx] # current halfEdgeIdx
#         (ss,se,sk) = face.halfEdges[idx+1] # successor halfEdgeIdx
#         assert (mySubdivision.graph[cs][ce][ck]['obj'].succIdx == (ss,se,sk))
        
#         (ts,te,tk) = mySubdivision.MDG[cs][ce][ck]['obj'].twinIdx
#         assert (ts == ss)

#     (cs,ce,ck) = face.halfEdges[-1] # current halfEdgeIdx
#     (ss,se,sk) = face.halfEdges[0] # successor halfEdgeIdx
#     assert (mySubdivision.graph[cs][ce][ck]['obj'].succIdx == (ss,se,sk))

#     (ts,te,tk) = mySubdivision.graph[cs][ce][ck]['obj'].twinIdx
#     assert (ts == ss)
# ########################################
