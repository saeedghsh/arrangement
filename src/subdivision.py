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

import operator

import numpy as np
import sympy as sym
import modifiedSympy as mSym
import networkx as nx

import multiprocessing as mp
import contextlib as ctx

import matplotlib.path as mpath
import matplotlib.transforms



import time
################################################################################
###################################################### parallelization functions
################################################################################
def intersection_star_(*args):
    global curves
    idx1, idx2 = args[0][0], args[0][1]
    return sym.intersection( curves[idx1].obj, curves[idx2].obj )

def intersection_star(*args):
    global curves
    idx1, idx2 = args[0][0], args[0][1]
    obj1 = curves[idx1].obj
    obj2 = curves[idx2].obj

    # Line-Line intersection - OK (speedwise)
    if isinstance(obj1, sym.Line) and isinstance(obj2, sym.Line):
        P1, P2 = obj1.p1 , obj1.p2
        P3, P4 = obj2.p1 , obj2.p2
        denom = (P1.x-P2.x)*(P3.y-P4.y) - (P1.y-P2.y)*(P3.x-P4.x)
        if np.abs(denom) > np.spacing(1):
            num_x = ((P1.x*P2.y)-(P1.y*P2.x))*(P3.x-P4.x) - (P1.x-P2.x)*((P3.x*P4.y)-(P3.y*P4.x))
            num_y = ((P1.x*P2.y)-(P1.y*P2.x))*(P3.y-P4.y) - (P1.y-P2.y)*((P3.x*P4.y)-(P3.y*P4.x))
            return [sym.Point(num_x/denom , num_y/denom)]
        else:
            return []

    else:
        # for arcs: check if intersections are in the interval
        intersections = sym.intersection( obj1, obj2 )
        for curve in [curves[idx1], curves[idx2]]:
            if isinstance(curve, mSym.ArcModified):
                for i in range(len(intersections)-1,-1,-1):
                    tval = curve.IPE(intersections[i])
                    if not(curve.t1 < tval < curve.t2):
                        intersections.pop(i)

        return intersections


################################################################################
def distance_star(*args):
    global intersectionPoints
    idx1, idx2 = args[0][0], args[0][1]

    p1 = intersectionPoints[idx1]
    p2 = intersectionPoints[idx2]
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    return sym.sqrt( (x1-x2)**2 + (y1-y2)**2 )

################################################################################
################################################################# object classes
################################################################# HalfEdge
################################################################# Face
################################################################################
class HalfEdge:
    def __init__ (self,
                  selfIdx, twinIdx,
                  cIdx, direction,
                  sTVal, eTVal):

        self.selfIdx = selfIdx   # self-index (startNodeIdx, endNodeIdx, pathIdx)
        self.twinIdx = twinIdx   # index to the twin half-edge
        self.succIdx = None # index to the successor half-edge

        # half edge Curve's attributes:
        self.cIdx = cIdx           # Index of the curve creating the edge
        self.direction = direction # defines the direction of t-value (positive: t2>t1, negative: t1>t2)

        # TODO: I think I should remove all the following    
        self.sTVal = sTVal
        self.eTVal = eTVal

        if self.sTVal < self.eTVal:
            assert (self.direction=='positive')
        elif self.sTVal > self.eTVal:
            assert (self.direction=='negative')            


################################################################################
class Face:
    def __init__(self, halfEdgeList, path):
        ''' '''
        self.halfEdges = halfEdgeList # a list of half-edges [(s,e,k), ...]
        self.path = path              # mpl.path
        self.holes = ()               # list of faces

    def get_area(self, considerHoles=True):
        '''
        Be aware that path.to_polygons() is an approximation of the face,
        if it contains curves, consequently the area would be approximated

        Green's theorem could provide an accurate measure of the area
        '''
        polygon = self.path.to_polygons()
        assert len(polygon) == 1
        x = polygon[0][:,0]
        y = polygon[0][:,1]
        PolyArea = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        
        holesArea = 0
        if considerHoles:
            for hole in self.holes:
                polygon = hole.path.to_polygons()
                assert len(polygon) == 1
                x = polygon[0][:,0]
                y = polygon[0][:,1]
                holesArea += 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

        return PolyArea - holesArea

    def is_point_inside(self, point):
        ''' '''
        if self.path.contains_point( (point.x,point.y) ):
            for hole in self.holes:
                if hole.path.contains_point( (point.x,point.y) ):
                    return False
            return True        
        return False

    def punch_hole(self, holeFace):
        '''
        although the path of the hole is suffiecent to detect inclusion of points,
        yet, holes are stored as faces, because the face instance of the hole is 
        required to able to traverse the boundary (half-edges) of the punched face.
        '''
       
        holes = list(self.holes)
        
        holeFace.holes = () # hole of hole is irrelevant
        holes.append(holeFace)

        # withholding nested holes
        redundant = []
        for idx1, h1 in enumerate(holes):
            for idx2, h2 in enumerate(holes):
                if idx1 != idx2 and h1.path.contains_path(h2.path):
                    redundant.append(idx2)

        for idx in sorted(redundant, reverse=True):
            holes.pop(idx)

        # storing final list of holes after conversion to tuple
        self.holes = tuple(holes)

    def get_punched_path(self):
        '''
        this is only useful for plotting
        the "path.contain_point()" doesn't work with holePunched pathes anyway

        no need to invert holes' trajectory,
        as they are supposedly superfaces, which means they have cw trajectory
        '''
        verts = self.path.vertices
        codes = self.path.codes

        for hole in self.holes:
            verts = np.append( verts, hole.path.vertices, axis=0)
            codes = np.append( codes, hole.path.codes)

        return mpath.Path(verts, codes)

        
################################################################################
class Decomposition:
    def __init__ (self, graph, faces, superFaceIdx=None):
        ''' '''
        self.graph = graph

        if superFaceIdx is not None:
            f = list(faces)
            self.superFace = f.pop(superFaceIdx)
            self.faces = tuple(f)
        else:
            self.superFace = None
            self.faces = faces

    def find_face(self, point):
        ''' '''
        for idx,face in enumerate(self.faces):
            if face.is_point_inside(point):
                return idx
        return None

    def find_neighbours(self, faceIdx):
        ''' '''
        MDG = self.graph

        # finding the indices to half-edges of the face
        twinsIdx= [ MDG[s][e][k]['obj'].twinIdx
                    for (s,e,k) in self.faces[faceIdx].halfEdges ]

        # finding the indices to half-edges of the holes of the face
        for hole in self.faces[faceIdx].holes:
            twinsIdx += [ MDG[s][e][k]['obj'].twinIdx
                          for (s,e,k) in hole.halfEdges ]
        
        # picking the face which have any of the twins as the boundary
        neighbours = []
        for fIdx,face in enumerate(self.faces):
            if any([twin in face.halfEdges for twin in twinsIdx]):
                neighbours.append(fIdx)
                
        # test
        assert( not (faceIdx in neighbours) )
    
        return neighbours
        
    def get_extents(self):
        ''' '''
        bboxes = [face.path.get_extents() for face in self.faces]
        return matplotlib.transforms.BboxBase.union(bboxes)
           


################################################################################
############################################################## Subdivision class
####################################################### aggregates from networkx
################################################################################
class Subdivision:

    '''
    '''

    ############################################################################
    def __init__ (self,curves , multiProcessing=0):
        '''
        curves are aggregated instances of sympy's geometric module
        (e.g. LineModified, CircleModified, ...)

        multiProcessing=0 -> no multi-processing
        multiProcessing=n -> n: number of processes
        '''
        self.multiProcessing = multiProcessing

        timing = False
        ########## reject duplicated curves and store internally
        self.curves = []
        self.store_curves(curves)

        ########## construct the base graph and subGraphs
        tic = time.time()
        self.MDG = nx.MultiDiGraph()
        if timing: print 'Graphs:', time.time() - tic

        #### STAGE A: construct nodes
        tic = time.time()
        self.construct_nodes()
        if timing: print 'nodes:', time.time() - tic

        #### STAGE B: construct edges
        tic = time.time()
        self.construct_edges()
        if timing: print 'edges:', time.time() - tic

        #### STAGE C: split the base graph into connected subgraphs
        tic = time.time()
        subGraphs = list(nx.connected_component_subgraphs(self.MDG.to_undirected()))
        subGraphs = [sg.to_directed() for sg in subGraphs]
        if timing: print 'connected components:', time.time() - tic

        ########## decomposition
        #### STAGE A: decomposition of each subgraph and merging
        tic = time.time()
        subDecompositions = []
        for iii, sg in enumerate( subGraphs ):
            faces = self.decompose_graph(sg)
            if len(faces) == 0:
                # we need to check if self.decompose_graph(sg) returns anything
                # for instance, if two line intersect only with each other,
                # there will be a subgraph of one node, with no edge or face
                # therefore, no decomposition shall be stored
                superFaceIdx = None

            else:
                # find the superFace of the decomposition                
                if len(faces) == 2:
                    # if only there are two faces, they will have the same area size
                    # so we look into the side attribute of half-edges of each face
                    # the one with negative side is selected as superFace

                    if len(faces[0].halfEdges) == 1:
                        # this is the case of a single circle
                        (s,e,k) = faces[0].halfEdges[0]
                        direction = self.MDG[s][e][k]['obj'].direction
                        superFaceIdx = 0 if direction=='negative' else 1

                    else:
                        # TODO: I have no idea whether this is correct! check!
                        # hypothesis:
                        # when a subdivision returns only two face, of which one is superFace
                        # the sum of the internal angles of superFace is always bigger than 
                        # the corresponding value of the inner face.
                        angleSum = [0,0]
                        for fIdx, face in enumerate(faces):
                            for (s,e,k) in face.halfEdges:                            
                                # ta: twin's departure angle
                                (ts,te,tk) = self.MDG[s][e][k]['obj'].twinIdx
                                twin = self.MDG[ts][te][tk]['obj']
                                curve = self.curves[twin.cIdx]
                                ta = curve.tangentAngle( self.MDG.node[ts]['point'],
                                                         twin.direction)
                                
                                # sa: successor's departure angle
                                (ss,se,sk) = self.MDG[s][e][k]['obj'].succIdx
                                succ = self.MDG[ss][se][sk]['obj']
                                curve = self.curves[succ.cIdx]
                                sa = curve.tangentAngle( self.MDG.node[ss]['point'],
                                                         succ.direction)
                                
                                # sa, ta in [0,2pi]
                                # and we want sth to be from ta to sa in ccw direction
                                sth = ta - sa if ta>sa else (ta+2*np.pi) - sa
                                angleSum[fIdx] += sth
                                
                        superFaceIdx = angleSum.index(max(angleSum))
                    

                    ########################################


                    # # TODO:
                    # # if it is desired not to allow a non-cyclic graph
                    # # be identified as a face, uncomment the follwing
                    # if faces[0].get_area() == 0:
                    #     superFaceIdx = None
                    #     faces = []


                else:
                    facesArea = [face.get_area() for face in faces]
                    superFaceIdx = facesArea.index(max(facesArea))
                    
            subDecompositions.append( Decomposition(sg, faces, superFaceIdx) )

        if timing: print 'decomposition:', time.time() - tic

        #### STAGE B: intersection of sub_decomposition
        tic = time.time()
        for idx1 in range(len(subDecompositions)):
            for idx2 in range(len(subDecompositions)):
                if idx1 != idx2:
                    sd1 = subDecompositions[idx1]
                    sd2 = subDecompositions[idx2]

                    # sd1 or sd2 could be empty 
                    if len(sd1.faces)>0 and len(sd2.faces)>0:

                        sampleNodeIdx = sd2.graph.nodes()[0]
                        samplePoint = sd2.graph.node[sampleNodeIdx]['point']

                        fIdx = sd1.find_face ( samplePoint )
                        if fIdx != None :
                            superFace = subDecompositions[idx2].superFace
                            subDecompositions[idx1].faces[fIdx].punch_hole ( superFace )

        self.subDecompositions = subDecompositions
        del subDecompositions
        if timing: print 'intersect graphs', time.time() - tic

        #### STAGE C: decomposition <- all together
        allFaces = ()
        for sd in self.subDecompositions:
            if sd:  # sd could be "None"
                allFaces += sd.faces
        self.decomposition = Decomposition(self.MDG, allFaces, superFaceIdx=None)
                       

    ############################################################################
    def store_curves(self, curves):
        # first discard duplicate and invalid curves
        # invalid curves are:    circles(radius <= 0)

        epsilon = np.spacing(10**10)

        for cIdx1 in range(len(curves)-1,-1,-1):
            obj1 = curves[cIdx1].obj
            obj1IsLine = isinstance(curves[cIdx1], (mSym.LineModified,
                                                    mSym.RayModified,
                                                    mSym.SegmentModified) )
            obj1IsCirc = isinstance(curves[cIdx1], mSym.CircleModified)
            obj1IsArc = isinstance(curves[cIdx1], mSym.ArcModified)
           
            if (obj1IsCirc or obj1IsArc) and curves[cIdx1].obj.radius<=0:
                # rejecting circles and arcs with (radius <= 0)
                curves.pop(cIdx1)

            else:
                # rejecting duplicated curves
                for cIdx2 in range(cIdx1):
                    obj2 = curves[cIdx2].obj
                    obj2IsLine = isinstance(curves[cIdx2], (mSym.LineModified,
                                                            mSym.RayModified,
                                                            mSym.SegmentModified) )
                    obj2IsCirc = isinstance(curves[cIdx2], mSym.CircleModified)
                    obj2IsArc = isinstance(curves[cIdx2], mSym.ArcModified)

                    if (obj1IsLine and obj2IsLine):
                        if sym.are_similar(curves[cIdx1].obj, curves[cIdx2].obj):
                            curves.pop(cIdx1)
                            break

                    elif (obj1IsCirc and obj2IsCirc):
                        dis = obj1.center.distance(obj2.center)
                        ris = obj1.radius - obj2.radius
                        if dis < epsilon and ris < epsilon:
                            curves.pop(cIdx1)
                            break

                    elif (obj1IsArc and obj2IsArc):
                        dis = obj1.center.distance(obj2.center)
                        ris = obj1.radius - obj2.radius
                        p1dis = curves[cIdx1].p1.distance(curves[cIdx2].p1)
                        p2dis = curves[cIdx1].p2.distance(curves[cIdx2].p2)
                        if dis < epsilon and ris < epsilon and p1dis < epsilon and p2dis < epsilon:
                            curves.pop(cIdx1)
                            break

        self.curves = curves


    ############################################################################
    def construct_nodes(self):
        '''
        |STAGE A| of Graph construction: node construction
        first we need a list of all intersections,
        while we can retrieve informations about each intersection point,
        such as which curves are intersecting at that point and what
        are the corresponding t-value of each curve's parametric expression
        at that intersection point.
        these informations are important to construct the nodes:

        intersectionsFlat ( <- intersections )
        ipsCurveIdx : curve indices of each ips/node
        ipsCurveTVal : ips/node's tValue over each assigned curve

        # step 1: finding all intersections
        # step 2: reject an intersection if it is not a point
        # step 3: handling non-intersecting curves
        # step 4: flattening the intersections [list-of-lists-of-lists] -> [list of lists]
        # step 5: adding two virtual intersection points at the -oo and +oo
        # step 6: find indeces of curves corresponding to each intersection point
        # step 7: merge collocated intersection points
        # step 8: find the t-value of each curve at the intersection
        # step 9: creating nodes from >intersection points<

        intersections
        this variable is a 2d matrix (list of lists) where each element at
        intersections[row][col] is itself a list of intersection points between
        two curves self.curves[row] and self.curves[col].
        '''

        intersections = [] # 2D array storage of intersection points

        # the indices to all following 3 lists are the same, i.e. ips_idx
        intersectionsFlat = []   # 1D array storage of intersection points
        ipsCurveIdx = []         # ipsCurveIdx[i]: idx of curves on nodes[i]
        ipsCurveTVal = []        # t-value of each node at assigned curves


        ########################################
        # step 1: finding all intersections
        intersections = [ [ []
                            for col in range(len(self.curves)) ]
                          for row in range(len(self.curves)) ]

        if self.multiProcessing: # with multiProcessing
            curvesTuplesIdx = [ [row,col]
                                for row in range(len(self.curves))
                                for col in range(row) ]

            global curves
            curves = self.curves
            with ctx.closing(mp.Pool(processes=self.multiProcessing)) as p:
                intersections_tmp = p.map( intersection_star, curvesTuplesIdx)
            del curves, p
            
            for (row,col),ips in zip (curvesTuplesIdx, intersections_tmp):
                intersections[row][col] = ips
                intersections[col][row] = ips
            del col,row, ips
                
        else:  # without multiProcessing
            for row in range(len(self.curves)):
                for col in range(row):
                    obj1 = self.curves[row].obj
                    obj2 = self.curves[col].obj
                    ip_tmp = sym.intersection(obj1,obj2)
                    intersections[row][col] = ip_tmp
                    intersections[col][row] = ip_tmp
            del col, row, ip_tmp

        ########################################
        # step 2: reject an intersection if it is not a point
        for row in range(len(self.curves)):
            for col in range(row):
                ips = intersections[row][col]
                if len(ips)>0 and isinstance(ips[0], sym.Point):
                    pass
                else:
                    intersections[row][col] = []
                    intersections[col][row] = []
        del col,row, ips

        ########################################
        # step 3: handling non-intersecting Curves
        # It would be problematic if a Curve does not intersect with any other
        # curves. The problem is that the construcion of the nodes relies on
        # the detection of the intersection points. Also the construction of
        # edges relies on the intersection points lying on each Curve.
        # No edge would result from a Curve without any intersection.
        # Consequently that Curves, not be presented by any edges, would the
        # be missed through the decomposition method (which relies on the edges)
        # >> This won't happen for unbounded curves (e.g. infinit
        # lines), since the -oo and +oo are assigned to them as intersection
        # points, however, this is a potential risk for bounded curves (e.g.
        # circles).        
        # >> To avoid this problem we can assign an arbitrary point on the
        # level-curve of the Curves, as a self-intersection point. This
        # self-intersection point won't mess the later stages even in case of
        # curves that already intesect with other curves. Hencefore for
        # the sake of simplicity, we add a self-intersection point to all
        # bounded curves.

        t = sym.Symbol('t')
        for row in range(len(self.curves)):
            if isinstance(self.curves[row].obj, sym.Circle):
                ips_n = np.sum( [ len(intersections[row][col])
                                  for col in range(len(self.curves)) ] )
                if ips_n==0:
                    p = self.curves[row].obj.arbitrary_point(t)
                    intersections[row][row] = [ p.subs([(t,0)]).evalf() ]

        ########################################
        # step 4: flattening the intersections list-of-lists-of-lists
        intersectionsFlat = [p
                             for row in range(len(self.curves))
                             for col in range(row+1) # for self-intersection
                             for p in intersections[row][col] ]

        ########################################
        # step 6: find indeces of curves corresponding to each intersection point
        ipsCurveIdx = [list(set([row,col]))
                       for row in range(len(self.curves))
                       for col in range(row+1) # for self-intersection 
                       for p in intersections[row][col] ]

        ########################################
        # step 7: merge collocated intersection points
        # duplicate: resulted of same curves intersection
        # collocated: resulted of different curves intersection

        if self.multiProcessing:
            distances = [ [ 0
                            for col in range(len(intersectionsFlat)) ]
                          for row in range(len(intersectionsFlat)) ]

            ipsTuplesIdx = [ [row,col]
                             for row in range(len(intersectionsFlat))
                             for col in range(row) ]

            global intersectionPoints
            intersectionPoints = intersectionsFlat
            with ctx.closing(mp.Pool(processes=self.multiProcessing)) as p:
                distancesFlat = p.map( distance_star, ipsTuplesIdx)
            del intersectionPoints
            
            for (row,col),dis in zip (ipsTuplesIdx, distancesFlat):
                dVal = dis.evalf()
                distances[row][col] = dVal
                distances[col][row] = dVal

            for idx1 in range(len(intersectionsFlat)-1,-1,-1):
                for idx2 in range(idx1):
                    if distances[idx1][idx2] < np.spacing(10**10): # == 0:
                        intersectionsFlat.pop(idx1)
                        s1 = set(ipsCurveIdx[idx1])
                        s2 = set(ipsCurveIdx[idx2])
                        ipsCurveIdx[idx2] = list(s1.union(s2)) 
                        ipsCurveIdx.pop(idx1)
                        break

        else:
            for idx1 in range(len(intersectionsFlat)-1,-1,-1):
                for idx2 in range(idx1):
                    if intersectionsFlat[idx1].distance(intersectionsFlat[idx2]) < np.spacing(10**10): # == 0:
                        s1 = set(ipsCurveIdx[idx2])
                        s2 = set(ipsCurveIdx[idx1])
                        ipsCurveIdx[idx2] = list(s1.union(s2)) 
                        ipsCurveIdx.pop(idx1)
                        intersectionsFlat.pop(idx1)
                        break
            # TODO: the distance_star is updated to reject invalid intersections
            # resulting from arcs, if multi-processing is not used, this loop
            # should be updated to reject those wrong intersections

        assert len(intersectionsFlat) == len(ipsCurveIdx)

        ########################################
        # step 8: find the t-value of each Curve at the intersection
        ipsCurveTVal = [ [ self.curves[cIdx].IPE(p) for cIdx in cIndices]
                            for (cIndices,p) in zip(ipsCurveIdx, intersectionsFlat) ]
        assert len(intersectionsFlat) == len(ipsCurveTVal)

        ########################################
        # step 9: creating nodes from >intersection points<
        '''
        pIdx: intersection point's index
        cIdx: intersecting curves' indices
        tVal: intersecting curves' t-value at the intersection point
        '''
        nodes = tuple( ( pIdx,
                         {'point': intersectionsFlat[pIdx],
                          'curveIdx': ipsCurveIdx[pIdx],
                          'curveTval':ipsCurveTVal[pIdx]} )
                       for pIdx in range(len(intersectionsFlat)) )

        self.MDG.add_nodes_from( nodes )
        assert len(self.MDG.nodes()) == len(intersectionsFlat)

    ############################################################################
    def construct_edges(self):
        '''
        |STAGE B| of Graph construction: edge construction
        to create edges, we need to list all the nodes
        located on each Curve, along with the t-value of the Curve
        at each node to sort nodes over the curve and segment the
        curve into edges according to the sorted nodes
        '''

        # TODO: merge step 1 and 2 into step 3

        ########################################
        # step 1: find intersection points of curves
        # indeces of intersection points corresponding to each curves
        curveIpsIdx = [[] for i in range(len(self.curves))]
        curveIpsTVal = [[] for i in range(len(self.curves))]

        for nodeIdx in self.MDG.nodes():
            for (tVal,cIdx) in zip(self.MDG.node[nodeIdx]['curveTval'],
                                   self.MDG.node[nodeIdx]['curveIdx']) :
                curveIpsIdx[cIdx].append(nodeIdx)
                curveIpsTVal[cIdx].append(tVal)

        ########################################
        # step 2: sort intersection points over curves
        # sorting iptersection points, according to corresponding tVal
        for cIdx in range(len(self.curves)):
            tmp = sorted(zip( curveIpsTVal[cIdx], curveIpsIdx[cIdx] ))
            curveIpsIdx[cIdx] = [pIdx for (tVal,pIdx) in tmp]
            curveIpsTVal[cIdx].sort()

        # ########################################
        # step 3: half-edge construction
        for (cIdx,curve) in enumerate(self.curves):

            ipsIdx = curveIpsIdx[cIdx]
            tvals = curveIpsTVal[cIdx]

            # step a:
            # for each curve, create all edges (half-edges) located on it
            if isinstance(curve.obj, ( sym.Line, sym.Segment, sym.Ray) ):

                startIdxList = ipsIdx[:-1]
                startTValList = tvals[:-1]

                endIdxList = ipsIdx[1:]
                endTValList = tvals[1:]


            elif isinstance(curve, mSym.ArcModified): # and isinstance(curve.obj, sym.Circ)

                # Important note: The order of elif matters...
                # isinstance(circle, mSym.ArcModified) - > False
                # isinstance(circle, mSym.CircleModified) - > True
                # isinstance(arc, mSym.ArcModified) - > True
                # isinstance(arc, mSym.CircleModified) - > True
                # this is why I first check the arc and then circle

                #TODO:  double-check
                startIdxList = ipsIdx[:-1]
                startTValList = tvals[:-1]

                endIdxList = ipsIdx[1:]
                endTValList = tvals[1:]
                
            elif isinstance(curve, mSym.CircleModified): # and isinstance(curve.obj, sym.Circle)
                # this case duplicaties the first point at the end of list

                startIdxList = ipsIdx
                startTValList = tvals

                endIdxList = ipsIdx[1:] + ipsIdx[:1]
                endTValList = tvals[1:] + tvals[:1]
                # fixing the singularity of the polar coordinate:
                # since all the ips and tVals are sorted [-pi , pi]
                # We only need to fix the last eTVal since
                # it is copied from the begining
                # idx = [idx for idx,n in enumerate(np.diff(endTValList)) if n<0]
                endTValList[-1] += 2*np.pi


            l = zip (startIdxList, startTValList, endIdxList, endTValList)

            # create a half-edge for each pair of start-end point
            for ( sIdx,sTVal, eIdx,eTVal ) in l:

                newPathKey1 = len(self.MDG[sIdx][eIdx]) if eIdx in self.MDG[sIdx].keys() else 0
                newPathKey2 = len(self.MDG[eIdx][sIdx]) if sIdx in self.MDG[eIdx].keys() else 0

                # in cases where sIdx==eIdx, twins will share the same key ==0
                # this will happen if there is only one node on a circle
                # also a non-intersecting circles with one dummy node
                # next line will take care of that only
                if sIdx==eIdx: newPathKey2 += 1
                
                idx1 = (sIdx, eIdx, newPathKey1)
                idx2 = (eIdx, sIdx, newPathKey2)

                # Halfedge(selfIdx, twinIdx, cIdx, side, sTVal, eTVal)

                # first half-edge
                direction = 'positive'                
                he1 = HalfEdge(idx1, idx2, cIdx, direction, sTVal, eTVal)
                e1 = ( sIdx, eIdx, {'obj':he1} )

                # second half-edge
                direction = 'negative'                
                he2 = HalfEdge(idx2, idx1, cIdx, direction, eTVal, sTVal)
                e2 = ( eIdx, sIdx, {'obj': he2} )

                self.MDG.add_edges_from([e1, e2])
    
    ############################################################################
    def get_all_HalfEdge_indices (self, graph=None):

        if graph==None: graph = self.MDG

        allHalfEdgeIdx = [(sIdx, eIdx, k)
                          for sIdx in graph.nodes()
                          for eIdx in graph.nodes()
                          if eIdx in graph[sIdx].keys() # if not, subd[sIdx][eIdx] is invalid
                          for k in graph[sIdx][eIdx].keys()]

        # TODO: isn't this better?
        # allHalfEdgeIdx = [(sIdx, eIdx, k)
        #                   for sIdx in graph.nodes()
        #                   for eIdx in graph[sIdx].keys()
        #                   for k in graph[sIdx][eIdx].keys()]

        return allHalfEdgeIdx

    ############################################################################
    def find_successor_HalfEdge(self, halfEdgeIdx, 
                              allHalfEdgeIdx=None,
                              direction='ccw_before'):

        # Note that in cases where there is a circle with one node on it,
        # the half-edge itself would be among candidates,
        # and it is important not to reject it from the candidates
        # otherwise the loop for find the face will never terminate!
        
        if allHalfEdgeIdx == None:
            allHalfEdgeIdx = self.get_all_HalfEdge_indices(self.MDG)

        (start, end, k) = halfEdgeIdx

        # "candidateEdges" are those edges starting from the "end" node
        candidateEdges = [idx
                          for (idx, heIdx) in enumerate(allHalfEdgeIdx)
                          if (heIdx[0] == end)]# and openList[idx] ]

        # reject the twin half-edge of the current half-edge from the "candidateEdges"
        twinIdx = self.MDG[start][end][k]['obj'].twinIdx
        for idx in range(len(candidateEdges)-1,-1,-1):
            if allHalfEdgeIdx[candidateEdges[idx]] == twinIdx:
                candidateEdges.pop(idx)
                break

        '''
        sorting keys:
        1st - Alpha = angle(1stDer) \in [0,2*pi)
        2nd - Beta  = curvature of the edge, considering the direction of it
        '''

        if len(candidateEdges) == 0:
            # if a followed path ends at an end point of a branch (not a cycle)
            # we have to let the pass return by the twin
            
            # TODO:
            # this unforunately will result in a having faces with null area
            # if a subgraph contains no cycle (i.e. a tree)

            return allHalfEdgeIdx.index(twinIdx)

        else:
            # reference: the 1st and 2nd derivatives of the twin half-edge
            (tStart, tEnd, tk) = twinIdx
            refObj = self.MDG[tStart][tEnd][tk]['obj']

            # sorting values of the reference (twin of the current half-edge)
            # 1stKey: alpha - 2ndkey: beta
            refObjCurve = self.curves[refObj.cIdx]
            sPoint = self.MDG.node[tStart]['point']
            refAlpha = refObjCurve.tangentAngle(sPoint, refObj.direction)
            refBeta = refObjCurve.curvature(sPoint, refObj.direction)

            # sorting values: candidates
            canAlpha = []
            canBeta = []
            for candidateIdx in candidateEdges:
                (cStart, cEnd, ck) = allHalfEdgeIdx[candidateIdx]
                canObj = self.MDG[cStart][cEnd][ck]['obj']
                canObjCurve = self.curves[canObj.cIdx]
                canAlpha.append( canObjCurve.tangentAngle(sPoint, canObj.direction) )
                canBeta.append( canObjCurve.curvature(sPoint, canObj.direction) )

            # sorting
            fullList  = zip( canAlpha, canBeta, candidateEdges )
            fullList += [(refAlpha,refBeta,'ref')]
            sortList  = sorted( fullList, key=operator.itemgetter(0, 1) )

            # picking the successor
            for i, (alpha,beta, idx) in enumerate(sortList):
                if idx == 'ref':
                    if direction=='ccw_before':
                        (a,c,successorIdx) = sortList[i-1]
                    elif direction=='ccw_after':
                        (a,c,successorIdx) = sortList[i+1]
                    break

            # # TODO: debugging - remove
            # ##############################
            # print '\n\tcurrent   half-edge:', halfEdgeIdx
            # for (alpha,beta, idx) in sortList:
            #     if idx == 'ref':
            #         print twinIdx, (alpha, beta), 'reference-twin'
            #     else:
            #         (s_, e_, k_) = allHalfEdgeIdx[idx]
            #         print (s_, e_, k_), (alpha, beta)
            # print '\tsuccessor   half-edge:', allHalfEdgeIdx[successorIdx]
            # ##############################
            return successorIdx


    ############################################################################
    def decompose_graph(self, graph):
        '''
        >>> face detection and identification procedure!
        '''

        # TODO:saesha - speeding up - not urgent
        # to find the successor to a half-edge, it's successor must be found.
        # to do so, a sorting of all half-edge is done around a node.
        # as it is now, the sorting of half-edges are done for every half-edge
        # but itsead I can sort half-edges around a node and cach all successors
        # this way I will perform sorting n x times, instead of e x times
        # n: number of nodes, e: numebr of half-edges
        # this requires moving the sorting procedure from find_successor_halfEdge
        # to a seperate method that caches all the successors in advance.
        
        faces = []
        allHalfEdgeIdx = self.get_all_HalfEdge_indices(graph)
        openList = [ 1 for heIdx in allHalfEdgeIdx]
        # note that the index to "openList" is equivalent to "allHalfEdgeIdx"
        
        ################################        
        # step three:
        # enumerate over the openList and identify faces
        while any(openList):

            # find the next item in the openList that is not closed (=0)
            openListIdx = next((i for i,v in enumerate(openList) if v != 0), None)

            if openList[openListIdx]: # not sure this is necessary!

                # initiate a new face with the first element of the openList
                face_tmp = [ allHalfEdgeIdx[openListIdx] ]

                # close the picked half-edge in the openList (set to 0)
                openList[openListIdx] = 0

                # find idx to starting and ending nodes of the picked half-edge
                sNodeIdx = allHalfEdgeIdx[openListIdx][0]
                eNodeIdx = allHalfEdgeIdx[openListIdx][1]

                # a trajectory following procedure, until a loop is closed.
                # through this while-loop, only "eNodeIdx" will update
                # the "sNodeIdx" will remain fix (i.e. for the current face)

                nextHalfEdgeIdx = self.find_successor_HalfEdge( face_tmp[0],
                                                                allHalfEdgeIdx,
                                                                direction='ccw_before')

                while face_tmp[0] != nextHalfEdgeIdx:#sNodeIdx == eNodeIdx:

                    # find the next half-edge in the trajectory
                    nextHalfEdgeIdx = self.find_successor_HalfEdge( face_tmp[-1],
                                                                    allHalfEdgeIdx,
                                                                    direction='ccw_before')

                    # update the face_tmp, if the next half-edge is open
                    if openList[nextHalfEdgeIdx]:
                        face_tmp.append( allHalfEdgeIdx[nextHalfEdgeIdx] )
                        eNodeIdx = allHalfEdgeIdx[nextHalfEdgeIdx][1]
                        openList[nextHalfEdgeIdx] = 0
                    else:
                        #print 'dump', face_tmp, allHalfEdgeIdx[nextHalfEdgeIdx], openList[nextHalfEdgeIdx]
                        break                        
                        # to be implemented later - or not!
                        # >> this will happen if one of the nodes is in infinity,
                        # which means we closed them in openList before.
                        # >> for now: ignore the face_tmp value,
                        # because it is the face that contains infinity!
                        # and the openList is updated properly so far

                # print face_tmp
                if sNodeIdx == eNodeIdx:
                    # with this condition we check if the face closed
                    # or the "while-loop" broke, reaching an infinity
                    # connected half edge.
                    faces.append(face_tmp)

                    # # TODO: debugging - remove
                    # ##############################
                    # print '\n\t the face:',face_tmp
                    # ##############################

                else:
                    pass

        ####### assign successor halfEdge Idx to each halfEdge:
        for edgeList in faces:
            for idx in range(len(edgeList)-1):
                (cs,ce,ck) = edgeList[idx] # current halfEdgeIdx
                (ss,se,sk) = edgeList[idx+1] # successor halfEdgeIdx
                self.MDG[cs][ce][ck]['obj'].succIdx = (ss,se,sk)
            (cs,ce,ck) = edgeList[-1] # current halfEdgeIdx
            (ss,se,sk) = edgeList[0] # successor halfEdgeIdx
            self.MDG[cs][ce][ck]['obj'].succIdx = (ss,se,sk)


        return tuple( Face( edgeList,
                            self.edgeList_2_mplPath(edgeList) )
                      for edgeList in faces )


    ################################### converting face to mpl.path
    def edgeList_2_mplPath (self, edgeList):

        # step1: initialization - openning the path
        (start, end, k) = edgeList[0]
        p = self.MDG.node[start]['point']
        x, y = p.x.evalf(), p.y.evalf()

        verts = [ (x,y) ]
        codes = [ mpath.Path.MOVETO ]

        # step2: construction - by following the trajectory of edges in edgeList
        for halfEdge in edgeList:

            (start, end, k) = halfEdge
            halfEdge_obj = self.MDG[start][end][k]['obj']
            cIdx = halfEdge_obj.cIdx
            sTVal = halfEdge_obj.sTVal
            eTVal = halfEdge_obj.eTVal

            # # TODO: eliminating sTVal and eTVal
            # sPoint = self.MDG.node[start]['point']
            # ePoint = self.MDG.node[end]['point']
            # sTVal = self.curves[cIdx].IPE(sPoint)
            # eTVal = self.curves[cIdx].IPE(ePoint)


            if isinstance(self.curves[cIdx].obj, ( sym.Line, sym.Segment, sym.Ray) ):
                p2 = self.MDG.node[end]['point']
                x, y = p2.x.evalf(), p2.y.evalf()
                verts.append( (x,y) )
                codes.append( mpath.Path.LINETO )

            elif isinstance(self.curves[cIdx].obj, sym.Circle):
                circ = self.curves[cIdx].obj
                xc, yc, rc = circ.center.x , circ.center.y , circ.radius

                # create an arc 
                t1 = np.float(sTVal) *(180 /np.pi)
                t2 = np.float(eTVal) *(180 /np.pi)

                if halfEdge_obj.direction == 'negative':
                    # TODO(saesha): which one?
                    arc = mpath.Path.arc( t2,t1 )
                    # arc = mpath.Path.arc( np.min([t1,t2]), np.max([t1,t2]) )
                else:
                    arc = mpath.Path.arc( t1,t2 )

                # transform arc
                transMat = matplotlib.transforms.Affine2D( )
                ## Note that the order of adding rot_scale_translate matters
                transMat.rotate(0) # rotate_around(x, y, theta)
                transMat.scale( rc ) # scale(sx, sy=None)
                transMat.translate(xc, yc) 
                arc = arc.transformed(transMat)

                vs = list( arc.vertices.copy() )
                cs = list( arc.codes.copy() )
                # TODO(saesha): which one?
                # cs[0] = mpath.Path.MOVETO
                # Lineto, because, otherwise, decompsong the face into polygones
                # for area approximation, it will result in disjoint segments
                cs[0] = mpath.Path.LINETO

                # reversing the order of vertices, if the halfEdge has negative direction
                if halfEdge_obj.direction == 'negative': vs.reverse()

                verts.extend( vs )
                codes.extend( cs )

        assert len(verts) == len(codes)

        # step3: finialize - closing the path
        # making sure that the last point of the path is not a control point of an arc
        if codes[-1] == 4:
            (start, end, k) = edgeList[0]
            p = self.MDG.node[start]['point']
            x, y = np.float(p.x.evalf()), np.float(p.y.evalf())
            verts.append( (x,y) )
            codes.append( mpath.Path.CLOSEPOLY )
        else:
            codes[-1] = mpath.Path.CLOSEPOLY 

        return mpath.Path(verts, codes)

    ############################################################################
    def save_to_image(self,  fileName, resolution=10.):
        ''' a color coded image of subdivision  '''
        pass #TODO(saesha)

    ############################################################################
    def update_with_new_functions(self, newFunctions=[]):
        ''' '''
        pass #TODO(saesha)
