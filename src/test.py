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

import unittest
import subdivision as sdv
import testCaseGenerator as tcg
reload(sdv)
reload(tcg)

class SubdivisionTests(unittest.TestCase):    
    '''
    '''

    def test_twinAssignment(self, subdiv):
        # checking the correctness of twin assignment in half-edge construction
        
        res = True
        for idx in subdiv.get_all_HalfEdge_indices():
            (s,e,k) = idx
            
            tidx = subdiv.MDG[s][e][k]['obj'].twinIdx
            (ts,te,tk) = tidx
            
            ttidx = subdiv.MDG[ts][te][tk]['obj'].twinIdx
            
            res = res and (idx == ttidx)

        return res


    def runTest(self):
        test_cases_key = [
            'default',                      # 0: 
            'default_small',                # 1: 
            'specialCase_01',               # 2: 4 tangent circles and a line
            'specialCase_02',               # 3: non-interrsecting circle
            'specialCase_03',               # 4: singel circle 4 intersections
            'specialCase_04',               # 5: 2 lines - one circle
            'specialCase_05',               # 6: a square - a circle
            'specialCase_06',               # 7: concentric circles
            'specialCase_07',               # 8: disjoint
            # 'intensive',                  # 9: 
            # 'intensive_x2',               # 10:
            # 'intensive_lineOnly',         # 11:
            # 'random',                     # 12:
            'star'                          # 13: 
        ]

        for key in test_cases_key:
            
            print 'testing case: '+key
            file_name = 'testCases/'+key+'.yaml'
            data = tcg.load_from_csv( file_name )
            curves = data['curves']
            subdiv = sdv.Subdivision(curves, multiProcessing=4)


            ########## testing twin assignment
            self.assertEqual( self.test_twinAssignment(subdiv), True,
                              'incorrect twin assignment')

            ########## testing number of nodes
            if 'number_of_nodes' in data.keys():
                n_nodes = data['number_of_nodes']
                self.assertEqual( len(subdiv.MDG.nodes()), n_nodes, 'incorrect number of nodes')
            else:
                print 'number of nodes is not available for ' + key, '...'

            ########## testing number of edges
            if 'number_of_edges' in data.keys():
                n_edges = data['number_of_edges']
                self.assertEqual( len(subdiv.MDG.edges()), n_edges,
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
                self.assertEqual( len(subdiv.subGraphs), n_subGraphs,
                                  'incorrect number of subGraphs')
            else:
                print 'number of subGraphs is not available for ' + key, '...'
            


if __name__ == '__main__':
    unittest.main()



