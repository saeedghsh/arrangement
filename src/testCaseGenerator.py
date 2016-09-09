
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

import sympy as sym
import numpy as np
import modifiedSympy as mSym
reload(mSym)


################################################################################
################################################################################
################################################################################
def get_test(name=None):
    
    Line = mSym.LineModified
    Circle = mSym.CircleModified
    Point = sym.Point
    
    if name == 'Default':
        curves = []
        curves += [ Line( args=(Point(xi,0), Point(xi,1)) )
                   for xi in np.linspace(0,10,10) ]
        curves += [ Line( args=(Point(0,yi), Point(1,yi)) )
                   for yi in np.linspace(0,10,10) ]
        curves += [ Circle( args=(Point(i,i), 2) )
                   for i in np.linspace(0,10,3) ]
        # Invalide Conic section Functions
        curves += [ Circle( args=(Point(0,0), 0) ) ]
        curves += [ Circle( args=(Point(0,0), -1) ) ]

        number_of_nodes = 132
        number_of_edges = 488
        number_of_faces = 113
        number_of_subGraphs = 1


    elif name == 'Default_Small' or not(name):
        curves = []
        curves += [ Line( args=(Point(xi,0), Point(xi,1)) )
                   for xi in np.linspace(0,2,4) ]
        curves += [ Line( args=(Point(0,yi), Point(1,yi)) )
                   for yi in np.linspace(0,2,4) ]
        curves += [ Circle( args=(Point(i,i), 2) )
                   for i in np.linspace(0,2,2) ]

        number_of_nodes = 36
        number_of_edges = 136
        number_of_faces = 33
        number_of_subGraphs = 1


    elif name == 'special_case_01':
        ''' 4 tangent circles and a tangent line'''
        curves = []
        curves += [ Line( args=(Point(0,yi), Point(1,yi)) )
                   for yi in [0]] # np.linspace(0,10,2) ]
        curves += [ Circle( args=(Point(0,i), np.abs(i)) )
                   for i in [-4, -2 , 2, 4] ]

        number_of_nodes = 1
        number_of_edges = 8
        number_of_faces = 4
        number_of_subGraphs = 1


    elif name == 'special_case_02':
        ''' non-interrsecting circle '''
        curves = []
        curves += [ Line( args=(Point(xi,0), Point(xi,1)) )
                   for xi in np.linspace(0,10,2) ]
        curves += [ Line( args=(Point(0,yi), Point(1,yi)) )
                   for yi in np.linspace(0,10,2) ]
        curves += [ Circle( args=(Point(5,5), 2) ) ]

        number_of_nodes = 5
        number_of_edges = 10
        number_of_faces = 2
        number_of_subGraphs = 2


    elif name == 'special_case_03':
        ''' singel circle 4 intersections '''
        curves = []
        curves += [ Line( args=(Point(xi,0), Point(xi,1)) )
                   for xi in np.linspace(0,10,2) ]
        curves += [ Line( args=(Point(0,yi), Point(1,yi)) )
                   for yi in np.linspace(0,10,2) ]
        curves += [ Circle( args=(Point(5,5), 5.) ) ]

        number_of_nodes = 8
        number_of_edges = 24
        number_of_faces = 5
        number_of_subGraphs = 1


    elif name == 'special_case_04':
        ''' 2 lines - one non-intersecting circle '''
        curves = []
        curves += [ Line( args=(Point(xi,0), Point(xi,1)) )
                   for xi in np.linspace(0,10,1) ]
        curves += [ Line( args=(Point(0,yi), Point(1,yi)) )
                   for yi in np.linspace(0,10,1) ]
        curves += [ Circle( args=(Point(5,5), 4.) ) ]

        number_of_nodes = 2
        number_of_edges = 2
        number_of_faces = 1
        number_of_subGraphs = 2


    elif name == 'special_case_05':
        ''' a square - a circle '''
        curves = []
        curves += [ Line( args=(Point(xi,0), Point(xi,1)) )
                   for xi in np.linspace(0,10,2) ]
        curves += [ Line( args=(Point(0,yi), Point(1,yi)) )
                   for yi in np.linspace(0,10,2) ]
        curves += [ Circle( args=(Point(-5,5), 4.) ) ]

        number_of_nodes = 5
        number_of_edges = 10
        number_of_faces = 2
        number_of_subGraphs = 2


    elif name == 'special_case_06': 
        ''' concentric circles '''
        curves = [ Circle( args=(Point(0,0), i) )
                  for i in range(1,5)]

        number_of_nodes = 4
        number_of_edges = 8
        number_of_faces = 4
        number_of_subGraphs = 4


    elif name == 'special_case_07':
        ''' disjoint'''
        curves = []
        curves += [ Line( args=(Point(xi,0), Point(xi,1)) )
                   for xi in np.linspace(0,10,4) ]
        curves += [ Line( args=(Point(0,yi), Point(1,yi)) )
                   for yi in np.linspace(0,10,2) ]
        curves += [ Circle( args=(Point(-5,5), 3) ) ]
        curves += [ Circle( args=(Point(-4,5), 3) ) ]

        number_of_nodes = 10
        number_of_edges = 28
        number_of_faces = 6
        number_of_subGraphs = 2


    elif name == 'intensive':
        curves = []
        curves += [ Line( args=(Point(xi,0), Point(xi,1)) )
                   for xi in np.linspace(0,10,10) ]
        curves += [ Line( args=(Point(0,yi), Point(1,yi)) )
                   for yi in np.linspace(0,10,10) ]
        curves += [ Circle( args=(Point(i,i), 3) )
                   for i in np.linspace(0,10,5) ]

        number_of_nodes = 196
        number_of_edges = 748
        number_of_faces = 177
        number_of_subGraphs = 1


    elif name == 'intensive_x2':
        curves = []
        curves += [ Line( args=(Point(xi,0), Point(xi,1)) )
                   for xi in np.linspace(0,10,30) ]
        curves += [ Line( args=(Point(0,yi), Point(1,yi)) )
                   for yi in np.linspace(0,10,30) ]
        curves += [ Circle( args=(Point(i,i), np.sqrt(20)/2) ) #np.abs(5-i))
                   for i in np.linspace(0,10,15) ]

        number_of_nodes = 0
        number_of_edges = 0
        number_of_faces = 0
        number_of_subGraphs = 0


    elif name == 'intensive_only_line':
        curves = []
        curves += [ Line( args=(Point(xi,0), Point(xi,1)) )
                   for xi in np.linspace(0,10,20) ]
        curves += [ Line( args=(Point(0,yi), Point(1,yi)) )
                   for yi in np.linspace(0,10,20) ]

        number_of_nodes = 400
        number_of_edges = 1520
        number_of_faces = 361
        number_of_subGraphs = 1


    elif name == 'star':
        a1 = 36 * np.pi/180
        a2 = 72 * np.pi/180
        p1 = Point(-np.sqrt(1 - np.cos(a1)**2 ), np.cos(a1))
        p2 = Point(+np.sqrt(1 - np.cos(a1)**2 ), np.cos(a1))
        p3 = Point(0, -1)
        curves = [ Line( args=(p1, 0) ),
                   Line( args=(p3, -np.tan(a1)) ),
                   Line( args=(p3, +np.tan(a1)) ),
                   Line( args=(p1, +np.tan(a2)) ),
                   Line( args=(p2, -np.tan(a2)) ),
                   Circle( args=(Point(0,0), np.cos(a1)) ),
                   Circle( args=(Point(0,0), 1) ),
                   Circle( args=(Point(0,0), 2) ) ]

        number_of_nodes = 25
        number_of_edges = 50
        number_of_faces = 26
        number_of_subGraphs = 1


    elif name == 'Random':
        ''' Random case'''
        curves = []
        nl = 4
        X1 = np.random.random(nl)
        Y1 = np.random.random(nl)
        X2 = np.random.random(nl)
        Y2 = np.random.random(nl)
        curves += [ Line( args=(Point(x1,y1), Point(x2,y2)) )
                   for (x1,y1,x2,y2) in zip(X1,Y1,X2,Y2) ]

        nc = 2
        Xc = np.random.random(nc)
        Yc = np.random.random(nc)
        Rc = np.random.random(nc) + .75
        curves += [ Circle( args=(Point(xc,yc), rc) )
                   for (xc,yc,rc) in zip(Xc,Yc,Rc) ]

        number_of_nodes = 0
        number_of_edges = 0
        number_of_faces = 0
        number_of_subGraphs = 0

    if name == 'example1':
        curves = []
        curves += [ Line( args=( Point(-1,-1), Point(1,1)) ) ]
        curves += [ Circle( args=(Point(2,0), 3) ) ]
        curves += [ Circle( args=(Point(-2,0), 3) ) ]

        number_of_nodes = 6
        number_of_edges = 22
        number_of_faces = 6
        number_of_subGraphs = 1

    return {'curves': curves,
            'number_of_nodes': number_of_nodes,
            'number_of_edges': number_of_edges,
            'number_of_faces': number_of_faces,
            'number_of_subGraphs': number_of_subGraphs}



# # for storing random cases  
# for f in curves:
#     if isinstance (f.obj, sym.Line):
#         x1,y1 , x2,y2 = np.float(f.obj.p1.x), np.float(f.obj.p1.y) , np.float(f.obj.p2.x), np.float(f.obj.p2.y)
#         print 'Line( args=(Point(' ,x1, ',' ,y1, '), Point(' ,x2, ',' ,y2, ')) )'
#     elif isinstance (f.obj, sym.Circle):
#         xc,yc,rc = np.float(f.obj.center.x), np.float(f.obj.center.y) , np.float(f.obj.radius)
#         print 'Circle( args=(Point(' ,xc, ',' ,yc, '), ',rc,') )'


# ####### Randome case 1
# curves = [Line( args=(Point(0.25, 0.05), Point(0.42, 0.83)) ),
#          Line( args=(Point(0.95, 0.70), Point(0.47, 0.14)) ),
#          Line( args=(Point(0.48, 0.81), Point(0.05, 0.58)) ),
#          Line( args=(Point(0.52, 0.60), Point(0.26, 0.80)) ),
#          Circle( args=(Point(0.25, 0.77), 1.06) ),
#          Circle( args=(Point(0.93, 0.41), 1.42) )]

# ####### Randome case 2
# curves = [ Line( args=(Point( 0.90, 0.24 ), Point( 0.02, 0.53 )) ),
#           Line( args=(Point( 0.46, 0.31 ), Point( 0.90, 0.27 )) ),
#           Line( args=(Point( 0.36, 0.35 ), Point( 0.90, 0.25 )) ),
#           Line( args=(Point( 0.51, 0.95 ), Point( 0.65, 0.35 )) ),
#           Circle( args=(Point( 0.48, 0.35 ),  1.55 ) ),
#           Circle( args=(Point( 0.33, 0.88 ),  1.47 ) ) ]
