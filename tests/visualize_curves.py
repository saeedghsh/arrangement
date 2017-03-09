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
import sys
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

if not( os.path.abspath('./../') in sys.path):
    sys.path.append( os.path.abspath('./../') )

from arrangement.utils import load_data_from_yaml
import arrangement.geometricTraits as trts
# reload(trts)

sys.path.append('/home/saesha/Dropbox/myGits/dev/')
import my_svg_parser_dev as mSVGp
reload(mSVGp)

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
    'example_10',                   # 26: 
    'example_11',                   # 27: 
    'caisr',                        # 28: 
    'specialCase_11',               # 29:
    'example_12',                   # 30: 3 circle holes in one square, + 1xArc
]

######################################## load data
timing = False

if 1:
    '''laoding yaml file'''    
    testNumber = 18
    file_name = 'testCases/'+test_cases_key[testNumber]+'.yaml'
    file_name = '/home/saesha/Dropbox/myGits/sample_data/HH/HIH_03.yaml'
    
else:    
    '''laoding svg file (+ converting 2 yaml)'''    
    svg_file_name = '/home/saesha/Dropbox/myGits/sample_image/svg_samples/circle_lines.svg'
    svg_file_name = '/home/saesha/Dropbox/myGits/sample_image/svg_samples/intel-01-occ-05cm.svg'
    svg_file_name = '/home/saesha/Dropbox/myGits/sample_image/svg_samples/svg_test_case_complete.svg'
    svg_file_name = '/home/saesha/Dropbox/myGits/sample_image/svg_samples/svg_test_case.svg'
    svg_file_name = '/home/saesha/Dropbox/myGits/sample_image/svg_samples/rect_circ_poly_line.svg'
    svg_file_name = '/home/saesha/Dropbox/myGits/sample_image/svg_samples/long_straight_path.svg'
    svg_file_name = '/home/saesha/Dropbox/myGits/sample_image/svg_samples/hfab_110-001-06-0501.svg'
    svg_file_name = '/home/saesha/Dropbox/myGits/arrangement/src/testCases/svg_files/islab_01.svg'
    svg_file_name = '/home/saesha/Dropbox/myGits/arrangement/src/testCases/svg_files/islab_01_with_image (copy).svg'
    
    # yaml_file_name = svg_to_ymal(dir_addr + svg_file_name)
    yaml_file_name = mSVGp.svg_to_ymal(svg_file_name)

    file_name = yaml_file_name
    print file_name

# laoding yaml file
data = load_data_from_yaml( file_name )
traits = data['traits']


### find a bounding box
x,y = [], []
for idx, trait in enumerate(traits):
    if isinstance( trait.obj, sym.Circle ):
        theta = np.linspace(0, 2*np.pi, 4, endpoint=False)
        xc,yc,rc = trait.obj.center.x, trait.obj.center.y, trait.obj.radius
        x.extend( xc + rc * np.cos(theta) )
        y.extend( yc + rc * np.sin(theta) )
    elif isinstance( trait.obj, sym.Line ):
        x.extend( [trait.obj.p1.x, trait.obj.p2.x] )
        y.extend( [trait.obj.p1.y, trait.obj.p2.y] )

if len(x)!=0:
    xMin = np.float(min([x_.evalf() for x_ in x]))
    xMax = np.float(max([x_.evalf() for x_ in x]))
    yMin = np.float(min([y_.evalf() for y_ in y]))
    yMax = np.float(max([y_.evalf() for y_ in y]))

    bLines  = [ sym.Line( (xMin,yMin),(xMax,yMin) ),
                sym.Line( (xMax,yMin),(xMax,yMax) ),
                sym.Line( (xMax,yMax),(xMin,yMax) ),
                sym.Line( (xMin,yMax),(xMin,yMin) ) ]


### plot
fig = plt.figure( figsize=(12, 12) )
axis = fig.add_subplot(111)

clrs = {'cir':'b', 'arc':'b', 'lin':'r', 'seg':'g', 'ray':'g'}
alph = {'cir': 1., 'arc': 1., 'lin': 1., 'seg': 1., 'ray': 1.}

for idx, trait in enumerate(traits):

    # note: order of the conditions matter since arcModified is subclass of CircleModified
    if isinstance( trait, trts.ArcModified ):
        t1,t2 = trait.t1 , trait.t2
        tStep = max( [np.float(np.abs(t2-t1)*(180/np.pi)) ,2])
        theta = np.linspace(np.float(t1), np.float(t2), tStep, endpoint=True)
        xc, yc, rc = trait.obj.center.x , trait.obj.center.y , trait.obj.radius
        x = xc + rc * np.cos(theta)
        y = yc + rc * np.sin(theta)
        axis.plot (x, y, clrs['arc'], alpha=alph['arc'])

    elif isinstance( trait, trts.CircleModified ):
        tStep = 360
        theta = np.linspace(0, 2*np.pi, tStep, endpoint=True)

        xc,yc,rc = trait.obj.center.x, trait.obj.center.y, trait.obj.radius
        x = xc + rc * np.cos(theta)
        y = yc + rc * np.sin(theta)
        axis.plot (x, y, clrs['cir'], alpha=alph['cir'])

    elif isinstance( trait, trts.SegmentModified ):
        x = [trait.obj.p1.x, trait.obj.p2.x]
        y = [trait.obj.p1.y, trait.obj.p2.y]
        axis.plot (x, y, clrs['seg'], alpha=alph['seg'])

    elif isinstance( trait, trts.RayModified ):
        # find the ending point on one of the bLines
        ips = []
        for bl in bLines:
            ips.extend( sym.intersection(trait.obj, bl) )
        for i in range(len(ips)-1,-1,-1):
            if not isinstance(ips[i], sym.Point):
                ips.pop(i)
            elif not ( (xMin <= ips[i].x <= xMax) and (yMin <= ips[i].y <= yMax) ):
                ips.pop(i)

        # plot the ray
        x = np.float(trait.obj.p1.x.evalf())
        y = np.float(trait.obj.p1.y.evalf())
        dx = np.float(ips[0].x.evalf()) - np.float(trait.obj.p1.x.evalf())
        dy = np.float(ips[0].y.evalf()) - np.float(trait.obj.p1.y.evalf())
        axis.arrow( np.float(x),np.float(y),
                    np.float(dx),np.float(dy), # shape='right',
                    # linewidth = 1, head_width = 0.1, head_length = 0.2,
                    fc = clrs['ray'], ec = clrs['ray'], alpha=alph['ray'])

    elif isinstance( trait, trts.LineModified ):
        # find the ending points on the bLines
        ips = []
        for bl in bLines:
            ips.extend( sym.intersection(trait.obj, bl) )

        for i in range(len(ips)-1,-1,-1):
            if not isinstance(ips[i], sym.Point):
                ips.pop(i)
            elif not ( (xMin <= ips[i].x <= xMax) and (yMin <= ips[i].y <= yMax) ):
                ips.pop(i)
                
        # plot the Line
        x = sorted( [np.float(ip.x.evalf()) for ip in ips] )
        y = sorted( [np.float(ip.y.evalf()) for ip in ips] )
        axis.plot (x, y, clrs['lin'], alpha=alph['lin'])

    else:
        print 'trait n#', str(idx), 'unknown'

plt.axis('equal')
plt.tight_layout()
plt.show()
