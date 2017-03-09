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
import skimage.transform
from . import geometricTraits as trts
# reload(trts)

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
                
        if 'boundary' in data.keys():
            result.update( {'boundary':  data['boundary']} )
        else:
            print ('\t WARNING boundary info not available in yaml WARNING ')

    result.update( {'traits': traits} )

    return result

################################################################################
def unbound_traits(trait_list):
    '''
    this method takes a list of traits and converts them as follow:
    ray -> line
    segment -> line
    line -> line
    arc -> circle
    circle -> circle
    '''

    for idx in range(len(trait_list)):

        # the trait before adjustment
        trait = trait_list[idx]
        
        if isinstance(trait, (trts.SegmentModified, trts.RayModified) ):
            # if the trait is (ray v segment) convert to line 
            trait = trts.LineModified( args=(trait.obj.p1,
                                             trait.obj.p2) )
            
        elif isinstance(trait, trts.ArcModified):
            # if the trait is (arc) convert to circle
            trait = trts.Circle( args=(trait.obj.center,
                                       trait.obj.radius) )
        
        else:
            # the trait is either line or circle
            # no need to adjust the trait
            pass

        # insertig the corrected trait back in the list
        trait_list[idx] = trait

    return trait_list

################################################################################
def bound_traits(trait_list, boundary):
    '''
    this method takes a list of traits and bounds them to boundary.
    line -> segment

    note:
    it only supports boundary [xMin, yMin, xMax, yMax] form. this means,
    the region of interest (ROI) is always rectangle (parallel to x/y axes)
    it could be extended to support any kind of region of interest by define
    the region with a path (e.g. matplotlib) and check if the intersection
    points are in bound.
    Since I don't need that genralization, I stick to this simple version
    '''
    xMin, yMin, xMax, yMax = boundary

    b_lines  = [ sym.Line( (xMin,yMin), (xMax,yMin) ),
                 sym.Line( (xMax,yMin), (xMax,yMax) ),
                 sym.Line( (xMax,yMax), (xMin,yMax) ),
                 sym.Line( (xMin,yMax), (xMin,yMin) ) ]



    for t_idx in range(len(trait_list)-1,-1,-1):

        # the trait before adjustment
        trait = trait_list[t_idx]
        
        if isinstance(trait, (trts.LineModified) ):            

            # finding all intersections between trait and boundary lines
            # the "if" condition is to reject lines:
            # sym.intersection would return a sym.Line if one of the traits
            # is the same as one of the boundary lines
            points = [ p
                       for b_line in b_lines
                       for p in sym.intersection( trait.obj, b_line )
                       if isinstance(p, sym.Point)]

            for p_idx in range(len(points)-1,-1,-1):
                # checking which points are out of bound wrt. RIO
                inbound_x = (xMin <= points[p_idx].x <= xMax)
                inbound_y = (yMin <= points[p_idx].y <= yMax)
                if not(inbound_x) or not(inbound_y):
                    points.pop(p_idx)
            
            if len(points)>2:
                # this means some points are coinciding on one corner
                for p_idx_1 in range(len(points)-1,-1,-1):
                    for p_idx_2 in range(p_idx_1): 
                        if points[p_idx_1].distance(points[p_idx_2])<np.spacing(10**10):
                            points.pop(p_idx_1)
                            break

            if len(points) == 2:
                # in case a trait does not pass through the RIO
                # insertig the corrected trait back in the list
                trait = trts.SegmentModified( args=(points[0],points[1]) )
                trait_list[t_idx] = trait
            else:
                trait_list.pop(t_idx)

        else:
            # the trait is either arc or circle
            # no need to adjust the trait
            pass

    return trait_list

################################################################################
def match_face_shape(face1, face2,
                     include_angle=False,
                     reject_size_mismatch=True):
    '''
    This method takes two faces and compares their shape.
    The comparison is a regex match by rolling (to left) the edge_type of the
    face1, and results a list of integers, each indicating steps of roll (to 
    left) that would match two faces.

    Input
    -----
    face1 (src), face2 (dst): instances of arrangement.face

    Parameter
    ---------
    include_angle: Boolean (default:False)
    reject_size_mismatch: Boolean (default:True)
    at the moment they are not functional

    Output
    ------
    matches: list of integer
    the number of steps of roll to left that would match two faces
    
    Note
    ----
    face1 and face2 do not belong to the same arrangement.
    since we don't need any information beyond face.attributes,
    no need to pass the arrangments.
    But the face attrribute['edge_type'] must be set before calling
    this method

    TODO:
    adopt the code to consider "include_angle"   
    '''

    print (' TODO: adopt the code to consider "include_angle"')   
    match = []

    attr1 = face1.attributes
    attr2 = face2.attributes

    edge_type1 = attr1['edge_type']
    edge_type2 = attr2['edge_type']
    
    if reject_size_mismatch and (len(edge_type1)!=len(edge_type2)):
        return match # returns an empty list

    for roll in range(len(edge_type1)):
        # rolling to left
        if edge_type2 == edge_type1[roll:]+edge_type1[:roll]:
            match += [roll]

    return match


################################################################################
def align_faces(arrangement1, arrangement2,
                f1Idx, f2Idx,
                tform_type='similarity' ):
    '''
    This method returns a list rigid transformation (+scale) that would aligne
    input faces.
    Matches from the "match_face_shape" method indicate rolling steps to the left.
    In numpy terms, it corresponds to np.roll(array, -roll_step, axis=0)

    Parameter
    ---------
    tform_type - {'similarity', 'affine', 'piecewise-affine', 'projective', 'polynomial'}
    types of transformation (default 'similarity')
    similarity: shape-preserving transformations
    affine: translation, scaling, homothety, reflection, rotation, shear mapping, and compositions
    see "print skimage.transform.estimate_transform.__doc__" for more details.

    Output
    ------
    alignments - dictionary
    Keys in the dictionary correspond to steps of rolls for match
    And the field to each key is the "GeometricTransform" object.
    see "print skimage.transform.estimate_transform.__doc__ for more details"

    Note
    ----
    src = face1 , dst = face2
    so the result is a transformation from face1 to face2    

    Note
    ----
    face1 and face2 do not belong to the same arrangement
    and since we need any node informations, we pass the arrangments too

    todo
    ----
    Could this happen?
    make sure sx == sy and sx>0
    note that sx <0 could happen, indicating a mirror
    for instance two quadrant of a circle could be aligned with rotation or flip

    '''

    alignments = {}
    f1 = arrangement1.decomposition.faces[f1Idx]
    f2 = arrangement2.decomposition.faces[f2Idx]
    matches = match_face_shape(f1,f2)    

    nodes1 = arrangement1.graph.node
    nodes2 = arrangement2.graph.node
    
    src = np.array( [(nodes1[n_idx]['obj'].point.x,
                      nodes1[n_idx]['obj'].point.y)
                     for n_idx in f1.attributes['edge_node_idx'] ],
                    dtype=np.float)

    dst = np.array( [(nodes2[n_idx]['obj'].point.x,
                      nodes2[n_idx]['obj'].point.y)
                     for n_idx in f2.attributes['edge_node_idx'] ],
                    dtype=np.float)

    if np.shape(src)[0] < 3:
        print( 'a face could have only one edge and one node! fix this ...' )
        return alignment # returning an empty list


    for roll in matches:
        tform = skimage.transform.estimate_transform( tform_type,
                                                      np.roll(src,-roll,axis=0),
                                                      dst )

        alignments[roll] = tform
    
    return alignments
