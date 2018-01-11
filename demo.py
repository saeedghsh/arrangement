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
import time
import arrangement.arrangement as arr
import arrangement.plotting as aplt

if __name__ == '__main__':
    ''' 
    list of supported parameters
    ----------------------------
    --multiprocessing

    example
    -------
    python demo.py --file_name 'tests/testCases/example_01.yaml' --multiprocessing 4
    '''

    args = sys.argv

    # fetching options from input arguments
    # options are marked with single dash
    options = []
    for arg in args[1:]:
        if len(arg)>1 and arg[0] == '-' and arg[1] != '-':
            options += [arg[1:]]

    # fetching parameters from input arguments
    # parameters are marked with double dash,
    # the value of a parameter is the next argument   
    listiterator = args[1:].__iter__()
    while 1:
        try:
            item = next( listiterator )
            if item[:2] == '--':
                exec(item[2:] + ' = next( listiterator )')
        except:
            break   

    # multiprocessing parameters (number of processes or False)
    multiprocessing = int(multiprocessing) if 'multiprocessing' in locals() else False

    # if input file is svg, convert to yaml and load yaml file
    if file_name.split('.')[-1] == 'svg':
        raise( NameError('SVG support is not available yet... Sorry') )
        file_name = arr.utls.svg_to_ymal(file_name, convert_segment_2_infinite_line=True)

    # load traits from yaml file
    data = arr.utls.load_data_from_yaml( file_name )
    traits = data['traits']

    # deploying arrangement (and decomposition)
    print( '\nstart decomposition:', file_name)

    tic = time.time()
    config = {'multi_processing':4, 'end_point':False, 'timing':False}
    arrange = arr.Arrangement(traits, config)
    print( 'Arrangement time: {:.5f}'.format( time.time() - tic) )

    # results
    print( 'nodes:\t\t {:d}'.format( len(arrange.graph.nodes()) ) )
    print( 'edges:\t\t {:d}'.format( len(arrange.graph.edges()) ) )
    print( 'faces:\t\t {:d}'.format( len(arrange.decomposition.faces) ) )
    print( 'subGraphs:\t {:d}'.format( len(arrange._subDecompositions) ) )

    aplt.animate_face_patches(arrange, timeInterval = .5* 1000)

