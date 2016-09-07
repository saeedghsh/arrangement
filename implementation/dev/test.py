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
        'star'                          # 13: 
    ]
    '''

    def runTest(self):
        test_cases_key = [
            'Default',                      # 0: 
            'Default_Small',                # 1: 
            # 'special_case_01',              # 2: 4 tangent circles and a line
            'special_case_02',              # 3: non-interrsecting circle
            'special_case_03',              # 4: singel circle 4 intersections
            'special_case_04',              # 5: 2 lines - one circle
            'special_case_05',              # 6: a square - a circle
            'special_case_06',              # 7: concentric circles
            'special_case_07',              # 8: disjoint
            # 'intensive',                    # 9: 
            # 'intensive_x2',                 # 10:
            # 'intensive_only_line',          # 11:
            # 'Random',                       # 12:
            'star'                          # 13: 
        ]

        for key in test_cases_key:

            print 'testing case: '+key
            test_case = tcg.get_test(name=key)
            curves = test_case['curves']
            n_nodes = test_case['number_of_nodes']
            n_edges = test_case['number_of_edges']
            n_faces = test_case['number_of_faces']
            n_subGraphs = test_case['number_of_subGraphs']
            
            subdiv = sdv.Subdivision(curves, multiProcessing=4)
            
            self.assertEqual( len(subdiv.MDG.nodes()), n_nodes, 'incorrect nodes')
            self.assertEqual( len(subdiv.MDG.edges()), n_edges, 'incorrect edges')
            self.assertEqual( len(subdiv.decomposition.faces), n_faces, 'incorrect faces')
            self.assertEqual( len(subdiv.subGraphs), n_subGraphs, 'incorrect subGraphs')


if __name__ == '__main__':
    unittest.main()
