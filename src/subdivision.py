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
import networkx as nx

import multiprocessing as mp
import contextlib as ctx

import matplotlib.path as mpath
import matplotlib.transforms

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

    obj1Line = isinstance(obj1, sym.Line)
    obj1Circ = isinstance(obj1, sym.Circle)
    # obj1Ray = isinstance(obj1, sym.Ray)

    obj2Line = isinstance(obj2, sym.Line)
    obj2Circ = isinstance(obj2, sym.Circle)
    # obj2Ray = isinstance(obj1, sym.Ray)


    # Line-Line intersection - OK (speedwise)
    if obj1Line and obj2Line: #(obj1Line or obj1Ray) and (obj2Line or obj2Ray):
        P1, P2 = obj1.p1 , obj1.p2
        P3, P4 = obj2.p1 , obj2.p2
        denom = (P1.x-P2.x)*(P3.y-P4.y) - (P1.y-P2.y)*(P3.x-P4.x)
        if np.abs(denom) > np.spacing(1):
            num_x = ((P1.x*P2.y)-(P1.y*P2.x))*(P3.x-P4.x) - (P1.x-P2.x)*((P3.x*P4.y)-(P3.y*P4.x))
            num_y = ((P1.x*P2.y)-(P1.y*P2.x))*(P3.y-P4.y) - (P1.y-P2.y)*((P3.x*P4.y)-(P3.y*P4.x))
            return [sym.Point(num_x/denom , num_y/denom)]
        else:
            return []

    # # Circle-Circle intersection - Not OK (speedwise)
    # elif obj1Circ and obj2Circ:
    #     # print 'circ-circ - start'
    #     c1, c2 = obj1.center, obj2.center
    #     r1, r2 = obj1.radius, obj2.radius
    #     d = sym.sqrt((c1.x-c2.x)**2 + (c1.y-c2.y)**2)

    #     if  d > r1+r2 : # no intersection
    #         # print '0a intersection'
    #         return []

    #     elif d < np.abs(r1-r2):  # no intersection
    #         # print '0b intersection'
    #         return []

    #     elif d == r1+r2:  # one intersection
    #         # print '1 intersection'
    #         a = (r1**2 - r2**2 + d**2) / (2*d) 
    #         # h = sym.sqrt(r1**2 - a**2)
    #         dx = a * (c2.x - c1.x) / d
    #         dy = a * (c2.y - c1.y) / d
    #         pMid = sym.Point(c1.x+dx , c1.y+dy)
    #         return [pMid]

    #     elif d < r1+r2:  # two intersections
    #         # print '2 intersections'
    #         a = (r1**2 - r2**2 + d**2) / (2*d) 
    #         h = sym.sqrt(r1**2 - a**2)
    #         dx = a * (c2.x - c1.x) / d
    #         dy = a * (c2.y - c1.y) / d
    #         pMid = [c1.x+dx , c1.y+dy]

    #         x3 = pMid[0] + h * ( c2.y - c1.y ) / d
    #         y3 = pMid[1] - h * ( c2.x - c1.x ) / d
    #         x4 = pMid[0] - h * ( c2.y - c1.y ) / d
    #         y4 = pMid[1] + h * ( c2.x - c1.x ) / d

    #         p3 = sym.Point(x3,y3)
    #         p4 = sym.Point(x4,y4)
    #         return [p3 ,p4]

    else:
        return sym.intersection( obj1, obj2 )


################################################################################
def distance_star_(*args):
    global intersectionPoints
    idx1, idx2 = args[0][0], args[0][1]
    return intersectionPoints[idx1].distance( intersectionPoints[idx2] )

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
################################################################# Node
################################################################# HalfEdge
################################################################# Face
################################################################################
class Curve:
    def __init__(self, curve,
                 itersectionPoints_Idx=(),
                 at_intersections_tValue=(),
                 at_intersections_derivative_1st=(),
                 at_intersections_derivative_2nd=() ):

        # the curve of the curve!
        self.curve = curve
        # indices of intersection point lying on the curve
        self.ipsIdx = itersectionPointsIdx
        self.ipsTVal = tValue_at_intersections
        self.ipsDer1st = at_intersections_derivative_1st
        self.ipsDer2nd = at_intersections_derivative_2nd

        assert len(self.ipsIdx) == len(self.ipsTVal)
        assert len(self.ipsIdx) == len(self.ipsDer1st)
        assert len(self.ipsIdx) == len(self.ipsDer2nd)

################################################################################
class Node:
    def __init__(self, point,
                 intersecting_curves_Idx=(),
                 intersecting_curves_Tval=()):
        
        self.point = point
        self.curveIdx = intersecting_curves_Idx
        self.curveTval = intersecting_curves_Tval

################################################################################
class HalfEdge:
    def __init__ (self,
                  selfIdx, twinIdx,
                  cIdx, side,
                  sIdx, sTVal, s1stDer, s2ndDer,
                  eIdx, eTVal, e1stDer, e2ndDer):

        self.selfIdx = selfIdx   # (sIdx, eIdx, pIdx)
        self.twinIdx = twinIdx   # twin half edge's index

        # half edge Curve's attributes:
        self.cIdx = cIdx         # Index of the curve creating the edge
        self.side = side         # defines the direction of t-value (in: t2>t1, out: t1>t2)
        
        # half edge attributes:
        self.sIdx = sIdx         # starting node's index        
        self.sTVal = sTVal
        self.s1stDer = s1stDer
        self.s2ndDer = s2ndDer

        self.eIdx = eIdx         # ending node's index
        self.eTVal = eTVal
        self.e1stDer = e1stDer
        self.e2ndDer = e2ndDer


################################################################################
class Face:
    def __init__(self, halfEdgeList, path):
        ''' '''
        self.halfEdges = halfEdgeList
        self.path = path
        self.holes = () # list of faces

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
        ''' '''
        holeFace.holes = () # withholding nested holes        
        self.holes += (holeFace,)


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
        
        self.graph = graph

        if superFaceIdx is not None:
            f = list(faces)
            self.superFace = f.pop(superFaceIdx)
            self.faces = tuple(f)
        else:
            self.superFace = None
            self.faces = faces

    def find_face(self, point):
        for idx,face in enumerate(self.faces):
            if face.is_point_inside(point):
                return idx
        return None

    def find_neighbours(self, faceIdx):
        return []
        
    def get_extents(self):
        bboxes = [face.path.get_extents() for face in self.faces]
        return matplotlib.transforms.BboxBase.union(bboxes)
           


################################################################################
############################################################## Subdivision class
####################################################### aggregated from networkx
################################################################################
class Subdivision:

    '''
    N: number of functions
    M: numebr of intersection points
    
    >>> intersections:
    all intersections stored in a list of lists of lists:
    fun1,  fun2, ...,  funN
    fun1:[[  [],    [], ...,    []],
    fun2: [  [],    [], ...,    []],
    ...                             
    funN: [  [],    [], ...,    []]]
    
    >>> ips:
    flat list of intersection points
    [p0, p1, ..., pM]
    
    >>> ipsFunIdx: [i.e. indices of functions of points]
    indeces of intersecting functions corresponding to ips
    each set_i is a set of indeces of functions resulting point_i
    [s0, s1, ..., sM], s_i = [idx | idx \in [0,N-1] ]
    
    >>> ipsFunTVal: [i.e. t-value of functions at points]
    t-value of each function at the intersection
    [tv0, tv1, ..., tvM], tv_i = [ f[idx].IPE(p_i) | idx \in s_i]
    
    >>> funIpsIdx: [i.e. indices of points over functions]
    indeces of intersection points corresponding to functions
    [s0, s1, ..., sN], s_i = [idx | idx \in [0,M-1] ]
    each set_i is a set of indeces of intersection points resulted from functions[i]
    
    >>> funIpsTVal: [i.e. functions t-value at each point]
    t-value of each function at the intersection
    [tv0, tv1, ..., tvN], tv_i = [f[idx].IPE(p_i) | idx \in s_i]
    '''
    # __slots__ = [ 'functions',
    #               'intersections',
    #               'ips',
    #               'ipsFunIdx',
    #               'ipsFunTVal',
    #               'funIpsIdx',
    #               'funIpsTVal' ]

    # __class__ = 'Subdivision'

    ############################################################################
    def __init__ (self,curves , multiProcessing=True):
        '''
        curves are aggregated instances of sympy's geometric module
        (e.g. LineModified, CircleModified)


        multiProcessing=0 -> no multi-processing
        multiProcessing=n -> n: number of processes
        '''
        self.multiProcessing = multiProcessing


        ########## reject duplicated curves and store internally
        self.curves = []
        self.store_curves(curves)

        ########## construct the base graph and subGraphs
        self.MDG = nx.MultiDiGraph()
        #### STAGE A: construct nodes
        self.construct_nodes()
        #### STAGE B: construct edges
        self.construct_edges()
        #### STAGE C: split the base graph into connected subgraphs
        subgraphs = list(nx.connected_component_subgraphs(self.MDG.to_undirected()))
        self.subGraphs = [sg.to_directed() for sg in subgraphs]
        del subgraphs

        ########## decomposition
        #### STAGE A: decomposition of each subgraph and merging
        subDecompositions = []
        for sg in self.subGraphs:
            faces = self.decompose_graph(sg)
            if faces:
                # we need to check if self.decompose_graph(sg) returns anything
                # for instance, if two line intersect only with each other,
                # there will be a subgraph of one node, with no edge or face
                # therefore, no decomposition
                facesArea = [face.get_area() for face in faces]
                superFaceIdx = facesArea.index(max(facesArea))
                subDecompositions.append( Decomposition(sg, faces, superFaceIdx) )
            else:
                subDecompositions.append( None )

        #### STAGE B: intersection of sub_decomposition
        for idx1 in range(len(subDecompositions)):
            for idx2 in range(len(subDecompositions)):
                if idx1 != idx2:
                    sd1 = subDecompositions[idx1]
                    sd2 = subDecompositions[idx2]
                    if sd1 and sd2: # sd1 or sd2 could be "None"
                        sampleNodeIdx = sd2.graph.nodes()[0]
                        samplePoint = self.intersectionsFlat[sampleNodeIdx]
                        fIdx = sd1.find_face ( samplePoint )
                        if fIdx != None :
                            superFace = subDecompositions[idx2].superFace
                            subDecompositions[idx1].faces[fIdx].punch_hole ( superFace )
        self.subDecompositions = subDecompositions
        del subDecompositions

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

        for cIdx1 in range(len(curves)-1,-1,-1):
            obj1 = curves[cIdx1].obj
            obj1IsLine = isinstance(obj1, sym.Line)
            obj1IsCirc = isinstance(obj1, sym.Circle)
            
            if obj1IsCirc and obj1.radius<=0:
                # rejecting circles with (radius <= 0)
                curves.pop(cIdx1)
            else:
                # rejecting duplicated curves
                for cIdx2 in range(cIdx1):
                    obj2 = curves[cIdx2].obj
                    obj2IsLine = isinstance(obj2, sym.Line)
                    obj2IsCirc = isinstance(obj2, sym.Circle)

                    if (obj1IsLine and obj2IsLine):
                        if sym.are_similar(obj1, obj2):
                            curves.pop(cIdx1)
                            break

                    elif (obj1IsCirc and obj2IsCirc):
                        dis = obj1.center.distance(obj2.center)
                        ris = obj1.radius - obj2.radius
                        if dis==0 and ris ==0:
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

        self.intersectionsFlat ( <- intersections )
        self.ipsCurveIdx
        self.ipsCurveTVal

        # step 1: finding all intersections
        # step 2: reject an intersection if it is not a point
        # step 3: handling non-intersecting curves
        # step 4: flattening the intersections [list-of-lists-of-lists] -> [list of lists]
        # step 5: adding two virtual intersection points at the -oo and +oo
        # step 6: find indeces of curves corresponding to each intersection point
        # step 7: merge collocated intersection points
        # step 8: find the t-value of each curve at the intersection
        # step 9: creating nodes from >intersection points<

        self.intersections
        this variable is a 2d matrix (list of lists) where each element at
        self.intersections[row][col] is itself a list of intersection points between
        two curves self.curves[row] and self.curves[col].

        self.standAloneCurvesIdx
        a list of indices to "self.curves" of non-intersecting curves        
        '''

        self.intersections = []
        self.standAloneCurvesIdx = [] # TODO: I actually never use this! should I keep it?

        self.intersectionsFlat = []
        self.ipsCurveIdx = []         # make Curve's internal and store all in nodes
        self.ipsCurveTVal = []        # make Curve's internal and store all in nodes

        self.nodes = [] 

        ########################################
        # step 1: finding all intersections
        self.intersections = [ [ []
                                 for col in range(len(self.curves)) ]
                               for row in range(len(self.curves)) ]

        if self.multiProcessing: # with multiProcessing
            curvesTuplesIdx = [ [row,col]
                                for row in range(len(self.curves))
                                for col in range(row) ]

            global curves
            curves = self.curves
            with ctx.closing(mp.Pool(processes=self.multiProcessing)) as p:
                intersections = p.map( intersection_star, curvesTuplesIdx)
            del curves, p
            
            for (row,col),ips in zip (curvesTuplesIdx, intersections):
                self.intersections[row][col] = ips
                self.intersections[col][row] = ips
            del col,row, ips
                
        else:  # without multiProcessing
            for row in range(len(self.curves)):
                for col in range(row):
                    obj1 = self.curves[row].obj
                    obj2 = self.curves[col].obj
                    ip_tmp = sym.intersection(obj1,obj2)
                    self.intersections[row][col] = ip_tmp
                    self.intersections[col][row] = ip_tmp
            del col, row, ip_tmp

        ########################################
        # step 2: reject an intersection if it is not a point
        for row in range(len(self.curves)):
            for col in range(row):
                ips = self.intersections[row][col]
                if len(ips)>0 and isinstance(ips[0], sym.Point):
                    pass
                else:
                    self.intersections[row][col] = []
                    self.intersections[col][row] = []
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
                ips_n = np.sum( [ len(self.intersections[row][col])
                                  for col in range(len(self.curves)) ] )
                if ips_n==0:
                    p = self.curves[row].obj.arbitrary_point(t)
                    self.intersections[row][row] = [ p.subs([(t,0)]).evalf() ]
                    self.standAloneCurvesIdx += [row]

        ########################################
        # step 4: flattening the intersections list-of-lists-of-lists
        self.intersectionsFlat = [p
                    for row in range(len(self.curves))
                    for col in range(row+1) # for self-intersection
                    for p in self.intersections[row][col] ]

        # ########################################
        # # step 5: adding two virtual intersection points at the -oo and +oo
        # # these will be used for handling unbounded regions (exterior faces)
        # self.intersectionsFlat.append(sym.Point(-sym.oo, -sym.oo))
        # self.intersectionsFlat.append(sym.Point(+sym.oo, +sym.oo))

        ########################################
        # step 6: find indeces of curves corresponding to each intersection point
        self.ipsCurveIdx = [list(set([row,col]))
                            for row in range(len(self.curves))
                            for col in range(row+1) # for self-intersection 
                            for p in self.intersections[row][col] ]

        # # step6_b: indexing all unbounded curves for infinity points
        # # lines are unbounded on both sides
        # lineIdx = [ idx  for idx, f in enumerate(self.curves)
        #             if isinstance(f.obj, sym.Line) ]
        # # assigning all line curves to both negative and positive infinity
        # self.ipsCurveIdx.append( lineIdx ) # for negative infinity
        # self.ipsCurveIdx.append( lineIdx ) # for positive infinity

        ########################################
        # step 7: merge collocated intersection points
        '''
        duplicate: resulted of same curves intersection
        collocated: resulted of different curves intersection
        '''
        if self.multiProcessing:
            distances = [ [ 0
                            for col in range(len(self.intersectionsFlat)) ]
                          for row in range(len(self.intersectionsFlat)) ]

            ipsTuplesIdx = [ [row,col]
                             for row in range(len(self.intersectionsFlat))
                             for col in range(row) ]

            global intersectionPoints
            intersectionPoints = self.intersectionsFlat
            with ctx.closing(mp.Pool(processes=self.multiProcessing)) as p:
                distancesFlat = p.map( distance_star, ipsTuplesIdx)
            del intersectionPoints
            
            for (row,col),dis in zip (ipsTuplesIdx, distancesFlat):
                dVal = dis.evalf()
                distances[row][col] = dVal
                distances[col][row] = dVal

            for idx1 in range(len(self.intersectionsFlat)-1,-1,-1):
                for idx2 in range(idx1):
                    if distances[idx1][idx2] < np.spacing(10**10): # == 0:
                        self.intersectionsFlat.pop(idx1)
                        s1 = set(self.ipsCurveIdx[idx1])
                        s2 = set(self.ipsCurveIdx[idx2])
                        self.ipsCurveIdx[idx2] = list(s1.union(s2)) 
                        self.ipsCurveIdx.pop(idx1)
                        break

        else:
            for idx1 in range(len(self.intersectionsFlat)-1,-1,-1):
                for idx2 in range(idx1):
                    if self.intersectionsFlat[idx1].distance( self.intersectionsFlat[idx2] ) == 0:
                        s1 = set(self.ipsCurveIdx[idx2])
                        s2 = set(self.ipsCurveIdx[idx1])
                        self.ipsCurveIdx[idx2] = list(s1.union(s2)) 
                        self.ipsCurveIdx.pop(idx1)
                        self.intersectionsFlat.pop(idx1)
                        break

        assert len(self.intersectionsFlat) == len(self.ipsCurveIdx)


        ########################################
        # step 8: find the t-value of each Curve at the intersection
        self.ipsCurveTVal = [ [ self.curves[cIdx].IPE(p) for cIdx in cIndices]
                            for (cIndices,p) in zip(self.ipsCurveIdx, self.intersectionsFlat) ]
        assert len(self.intersectionsFlat) == len(self.ipsCurveTVal)

        ########################################
        # step 9: creating nodes from >intersection points<
        '''
        pIdx: intersection point's index
        cIdx: intersecting curves' indices
        tVal: intersecting curves' t-value at the intersection point
        '''
        nodes = tuple( (pIdx, { 'obj': Node(self.intersectionsFlat[pIdx],
                                       self.ipsCurveIdx[pIdx],
                                       self.ipsCurveTVal[pIdx])} )
                  for pIdx in range(len(self.intersectionsFlat)) )

        self.nodes = nodes
        self.MDG.add_nodes_from( nodes )
        assert len(self.MDG.nodes()) == len(self.intersectionsFlat)

    ############################################################################
    def construct_edges(self):
        '''
        |STAGE B| of Graph construction: edge construction
        to create edges, we need to list all the intersection points
        located on each Curve, along with the t-value of the Curve
        at each intersection point

        self.curveIpsIdx
        self.curveIpsTVal
        '''
        self.edges = ()
        self.curveIpsIdx = []
        self.curveIpsTVal = []

        ########################################
        # step 1: find intersection points of curves
        # indeces of intersection points corresponding to each curves
        self.curveIpsIdx = [[] for i in range(len(self.curves))]
        self.curveIpsTVal = [[] for i in range(len(self.curves))]
        for pIdx in range(len(self.intersectionsFlat)):
            for (tVal,cIdx) in zip(self.ipsCurveTVal[pIdx], self.ipsCurveIdx[pIdx]) :
                self.curveIpsIdx[cIdx].append(pIdx)
                self.curveIpsTVal[cIdx].append(tVal)

        ########################################
        # step 2: sort intersection points over curves
        # sorting iptersection points, according to corresponding tVal
        for cIdx in range(len(self.curves)):
            # print 'on Curve ',cIdx,':', self.curveIpsTVal[cIdx], self.curveIpsIdx[cIdx]
            tmp = sorted(zip( self.curveIpsTVal[cIdx], self.curveIpsIdx[cIdx] ))
            self.curveIpsIdx[cIdx] = [pIdx for (tVal,pIdx) in tmp]
            self.curveIpsTVal[cIdx].sort()

        # ########################################
        # # step 3: derivatives over curves
        # self.curveDer1st = [[] for i in range(len(self.curves))]
        # self.curveDer2nd = [[] for i in range(len(self.curves))]
        # for (cIdx,f) in enumerate(self.curves):
        #     self.curveDer1st[cIdx] = [ f.firstDerivative(self.intersectionsFlat[pIdx])
        #                              for pIdx in self.curveIpsIdx[cIdx] ]
        #     self.curveDer2nd[cIdx] = [ f.secondDerivative(self.intersectionsFlat[pIdx])
        #                              for pIdx in self.curveIpsIdx[cIdx] ]

        # ########################################
        # step 4: half-edge construction
        for (cIdx,c) in enumerate(self.curves):

            # step a:
            # for each curve, creat all edge (half-edges) located on it
            if isinstance(c.obj, sym.Line):
                ipsIdx = self.curveIpsIdx[cIdx]
                tvals = self.curveIpsTVal[cIdx]

                startIdxList = ipsIdx[:-1]
                startTValList = tvals[:-1]

                endIdxList = ipsIdx[1:]
                endTValList = tvals[1:]

            elif isinstance(c.obj, sym.Circle):
                # this case duplicaties the first point at the end of list
                ipsIdx = self.curveIpsIdx[cIdx]
                tvals = self.curveIpsTVal[cIdx]

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

            for ( sIdx,sTVal, eIdx,eTVal ) in l:

                newPathKey1 = len(self.MDG[sIdx][eIdx]) if eIdx in self.MDG[sIdx].keys() else 0
                newPathKey2 = len(self.MDG[eIdx][sIdx]) if sIdx in self.MDG[eIdx].keys() else 0

                # in cases where sIdx==eIdx, twins will share the same key ==0
                # this will happen if there is only one node on a circle
                # ( also a non-intersecting circles with one dummy node)
                # next line will take care of that only
                # if cIdx in self.standAloneCurvesIdx: newPathKey2 += 1
                if sIdx==eIdx: newPathKey2 += 1

                
                idx1 = (sIdx, eIdx, newPathKey1)
                idx2 = (eIdx, sIdx, newPathKey2)

                # Halfedge(selfIdx, twinIdx,
                #          cIdx, side,
                #          sIdx, sTVal, s1stDer, s2ndDer,
                #          eIdx, eTVal, e1stDer, e2ndDer)

                ps = self.intersectionsFlat[sIdx]
                pe = self.intersectionsFlat[eIdx]
                c = self.curves[cIdx]

                # first half-edge
                direction = 'positive'                
                s1stDer = c.firstDerivative(ps, direction)
                s2ndDer = c.secondDerivative(ps, direction)
                e1stDer = c.firstDerivative(pe, direction)
                e2ndDer = c.secondDerivative(pe, direction)
                he1 = HalfEdge(idx1, idx2, cIdx, direction,
                               sIdx, sTVal, s1stDer, s2ndDer,
                               eIdx, eTVal, e1stDer, e2ndDer)
                e1 = ( sIdx, eIdx, {'obj':he1} )

                # second half-edge
                direction = 'negative'                
                s1stDer = c.firstDerivative(ps, direction)
                s2ndDer = c.secondDerivative(ps, direction)
                e1stDer = c.firstDerivative(pe, direction)
                e2ndDer = c.secondDerivative(pe, direction)
                he2 = HalfEdge(idx2, idx1, cIdx, direction,
                               eIdx, eTVal, e1stDer, e2ndDer,
                               sIdx, sTVal, s1stDer, s2ndDer)
                e2 = ( eIdx, sIdx, {'obj': he2} )

                self.MDG.add_edges_from([e1, e2])
                self.edges = self.edges + (e1, e2)

    
    ############################################################################
    def get_all_HalfEdge_indices (self, graph=None):

        if graph==None: graph = self.MDG

        allHalfEdgeIdx = [(sIdx, eIdx, k)
                          for sIdx in graph.nodes()
                          for eIdx in graph.nodes()
                          if eIdx in graph[sIdx].keys() # if not, subd[sIdx][eIdx] is invalid
                          for k in graph[sIdx][eIdx].keys()]
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

        # reference: the 1st and 2nd derivatives of the twin half-edge
        (tStart, tEnd, tk) = twinIdx
        refObj = self.MDG[tStart][tEnd][tk]['obj']


        # sorting values: reference
        (dx,dy) = refObj.s1stDer
        # 1stKey:
        refAlpha = np.arctan2(dy,dx)
        refAlpha = np.mod(refAlpha + 2*np.pi , 2*np.pi)
        # 2ndkey:
        refBeta = self.curves[refObj.cIdx].curvature(direction=refObj.side)

        # sorting values: candidates
        canAlpha = []
        canBeta = []

        for candidateIdx in candidateEdges:
            (cStart, cEnd, ck) = allHalfEdgeIdx[candidateIdx]
            canObj = self.MDG[cStart][cEnd][ck]['obj']

            (dx,dy) = canObj.s1stDer
            canAlpha.append( np.mod( np.arctan2(dy, dx) + 2*np.pi , 2*np.pi) )
            canBeta.append( self.curves[canObj.cIdx].curvature(direction=canObj.side) )


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

        faces = []
        allHalfEdgeIdx = self.get_all_HalfEdge_indices(graph)
        openList = [ 1 for heIdx in allHalfEdgeIdx]
        # note that the index to "openList" is equivalent to "allHalfEdgeIdx"

        # ################################        
        # # step one:
        # # put all edges conndected to infinity in the
        # # closed loop without creating a face from them 
        # pInfIdx = len(self.intersectionsFlat)-1
        # nInfIdx = len(self.intersectionsFlat)-2

        # for (openListIdx, isOpen) in enumerate(openList):
        #     if isOpen:
        #         sNodeIdx = allHalfEdgeIdx[openListIdx][0]
        #         eNodeIdx = allHalfEdgeIdx[openListIdx][1]

        #         c1 = (sNodeIdx==pInfIdx)
        #         c2 = (sNodeIdx==nInfIdx)
        #         c3 = (eNodeIdx==pInfIdx)
        #         c4 = (eNodeIdx==nInfIdx)
                
        #         if  c1 or c2 or c3 or c4:
        #             # print 'closing edge: ', self.allHalfEdgeIdx[openListIdx]
        #             openList[openListIdx] = 0

        # ################################        
        # # step two: 
        # # put the stand alone curves in the closed list
        # for (openListIdx, isOpen) in enumerate(openList):
        #     if isOpen:
        #         start, end, k = allHalfEdgeIdx[openListIdx]
        #         cIdx = graph[start][end][k]['obj'].cIdx
        #         if cIdx in self.standAloneCurvesIdx:
        #             openList[openListIdx] = 0
        
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

                          
        return tuple( Face( edgeList, self.edgeList_2_mplPath(edgeList) )
                 for edgeList in faces )


    ################################### converting face to mpl.path
    def edgeList_2_mplPath (self, edgeList):

        # step1: initialization - openning the path
        (start, end, k) = edgeList[0]
        p = self.intersectionsFlat[start]
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

            if isinstance(self.curves[cIdx].obj, sym.Line):
                p2 = self.intersectionsFlat[end]
                x, y = p2.x.evalf(), p2.y.evalf()
                verts.append( (x,y) )
                codes.append( mpath.Path.LINETO )

            elif isinstance(self.curves[cIdx].obj, sym.Circle):
                circ = self.curves[cIdx].obj
                xc, yc, rc = circ.center.x , circ.center.y , circ.radius

                # create an arc 
                t1 = np.float(sTVal) *(180 /np.pi)
                t2 = np.float(eTVal) *(180 /np.pi)

                if halfEdge_obj.side == 'negative':
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
                if halfEdge_obj.side == 'negative': vs.reverse()

                verts.extend( vs )
                codes.extend( cs )

        assert len(verts) == len(codes)

        # step3: finialize - closing the path
        # make sure that the last point of the path is not a control point of an arc
        if codes[-1] == 4:
            (start, end, k) = edgeList[0]
            p = self.intersectionsFlat[start]
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
