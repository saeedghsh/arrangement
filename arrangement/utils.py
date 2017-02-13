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

import sys
if sys.version_info[0] == 3:
    from importlib import reload
elif sys.version_info[0] == 2:
    pass


import yaml
import sympy as sym
import numpy as np
from . import geometricTraits as trts
reload(trts)

################################################################################
################################################################################
################################################################################
def load_data_from_yaml(fileName=None):

    Point = sym.Point
    Line = trts.LineModified
    Ray = trts.RayModified
    Segment = trts.SegmentModified
    Circle = trts.CircleModified
    Arc = trts.ArcModified

    result = {}
    traits = []

    if 'random' in fileName: # generate an almost random case 

        result.update( {'dataset': 'random'} )

        nl = 4
        X1 = np.random.random(nl)
        Y1 = np.random.random(nl)
        X2 = np.random.random(nl)
        Y2 = np.random.random(nl)
        traits += [ Line( args=(Point(x1,y1), Point(x2,y2)) )
                   for (x1,y1,x2,y2) in zip(X1,Y1,X2,Y2) ]

        nc = 2
        Xc = np.random.random(nc)
        Yc = np.random.random(nc)
        Rc = np.random.random(nc) + .75
        traits += [ Circle( args=(Point(xc,yc), rc) )
                   for (xc,yc,rc) in zip(Xc,Yc,Rc) ]



    elif 'star' in fileName:

        result.update( {'dataset': 'star'} )

        # generate the star case
        # happens to be very tricky
        a1 = 36 * np.pi/180
        a2 = 72 * np.pi/180
        p1 = Point(-np.sqrt(1 - np.cos(a1)**2 ), np.cos(a1))
        p2 = Point(+np.sqrt(1 - np.cos(a1)**2 ), np.cos(a1))
        p3 = Point(0, -1)
        traits = [ Line( args=(p1, 0) ),
                   Line( args=(p3, -np.tan(a1)) ),
                   Line( args=(p3, +np.tan(a1)) ),
                   Line( args=(p1, +np.tan(a2)) ),
                   Line( args=(p2, -np.tan(a2)) ),
                   Circle( args=(Point(0,0), np.cos(a1)) ),
                   Circle( args=(Point(0,0), 1) ),
                   Circle( args=(Point(0,0), 2) ) ]        

        result.update( {'number_of_nodes': 25} )
        result.update( {'number_of_edges': 100} )
        result.update( {'number_of_faces': 26} )
        result.update( {'number_of_subGraphs': 1} )       

    else: 

        stream = open(fileName, 'r')
        data = yaml.load(stream)

        if 'dataset' in data.keys():
            result.update( {'dataset': data['dataset']} )

        if 'testValues' in data.keys():
            result.update( data['testValues'][0] )

        if 'lines' in data.keys():
            for l in data['lines']:
                if len(l) == 4: #[x1,y1,x2,y2]
                    traits += [ Line( args=(Point(l[0],l[1]), Point(l[2],l[3]))) ]
                elif len(l) == 3: #[x1,y1,slope]
                    traits += [ Line( args=(Point(l[0],l[1]), l[2])) ]

        if 'segments' in data.keys():
            for s in data['segments']:
                seg = Segment( args=(Point(s[0],s[1]), Point(s[2],s[3])))
                if isinstance(seg.obj, sym.Segment): # not(isinstance(seg.obj, sym.Point))
                    traits += [ seg ]

        if 'rays' in data.keys():
            for r in data['rays']:
                traits += [ Ray( args=(Point(r[0],r[1]), Point(r[2],r[3]))) ]


        if 'circles' in data.keys():
            for c in data['circles']:
                traits += [ Circle( args=(Point(c[0],c[1]), c[2]) ) ]

        if 'arcs' in data.keys():
            for a in data['arcs']:
                traits += [ Arc( args=( Point(a[0],a[1]), a[2], (a[3],a[4])) ) ]

    result.update( {'traits': traits} )

    return result
