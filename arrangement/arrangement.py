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

from __future__ import print_function, division

import time
import operator
import itertools 

import numpy as np
import sympy as sym
import networkx as nx

import multiprocessing as mp
import contextlib as ctx

import matplotlib.path as mpath
import matplotlib.transforms

from . import geometricTraits as trts
from . import utils as utls

# import svgpathtools

################################################################################
###################################################### parallelization functions
################################################################################
def intersection_star(*args):
    global traits
    idx1, idx2 = args[0][0], args[0][1]
    obj1 = traits[idx1].obj
    obj2 = traits[idx2].obj

    intersections = sym.intersection( obj1, obj2 )
    # for arcs: check if intersections are in the interval
    for trait in [traits[idx1], traits[idx2]]:
        if isinstance(trait, trts.ArcModified):
            for i in range(len(intersections)-1,-1,-1):
                tval = trait.IPE(intersections[i])
                conditions = [ trait.t1 < tval < trait.t2 ,
                               trait.t1 < tval+2*np.pi < trait.t2, 
                               trait.t1 < tval-2*np.pi < trait.t2 ]
                # if (trait.t1 < tval < trait.t2):
                if any(conditions):
                    pass
                else:
                    intersections.pop(i)

    return intersections

################################################################################
###################################################### some other functions
################################################################################
################################### converting face to mpl.path
def edgeList_2_mplPath (edgeList, graph, traits):
    '''
    important note:
    this works only if the edgeList is sorted
    and the sequence of edges shall represent a simple closed trait (path)
    '''
    
    # step1: initialization - openning the path
    (start, end, k) = edgeList[0]
    p = graph.node[start]['obj'].point
    x, y = p.x.evalf(), p.y.evalf()
    
    verts = [ (x,y) ]
    codes = [ mpath.Path.MOVETO ]

    # step2: construction - by following the trajectory of edges in edgeList
    for halfEdge in edgeList:

        (start, end, k) = halfEdge
        halfEdge_obj = graph[start][end][k]['obj']
        cIdx = halfEdge_obj.traitIdx
        sTVal, eTVal = halfEdge_obj.get_tvals(traits, graph.node)


        if isinstance(traits[cIdx].obj, ( sym.Line, sym.Segment, sym.Ray) ):
            p2 = graph.node[end]['obj'].point
            x, y = p2.x.evalf(), p2.y.evalf()
            verts.append( (x,y) )
            codes.append( mpath.Path.LINETO )

        elif isinstance(traits[cIdx].obj, sym.Circle):
            circ = traits[cIdx].obj
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
            # which one? cs[0] = mpath.Path.MOVETO or mpath.Path.LINETO
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
        p = graph.node[start]['obj'].point
        x, y = np.float(p.x.evalf()), np.float(p.y.evalf())
        verts.append( (x,y) )
        codes.append( mpath.Path.CLOSEPOLY )
    else:
        codes[-1] = mpath.Path.CLOSEPOLY 
        
    return mpath.Path(verts, codes)


################################################################################
################################################################# object classes
################################################################# HalfEdge
################################################################# Face
################################################################################
class Node:
    def __init__ (self, selfIdx, point, traitIdx, traits):
        '''
        '''

        self.attributes = {}
        self.selfIdx = selfIdx   # self-index
        self.point = point
        self._traitIdx = traitIdx
        self._traitTval = []
        self.update_tval(traits)


    ####################################
    def transform_sequence(self, operTypes, operVals, operRefs, traits):
        '''
        Node class

        this method performs a sequence of transformation processes expressed by
        
        * operTypes: defines the type of each transformation
        * operVals: the values for each transformation
        * operRefs: the reference point for each transformation
        -- reference point is irrelevant for translation, still should be provided for consistency
        
        example:
        obj.transform_sequence( operTypes='TTRST',
        operVals=( (.5,-.5), (2,0), np.pi/2, (.5,.5), (3,-1) ),
        operRefs=( (0,0),    (0,0), (2,2),   (0,0),   (0,0)  ) )
        
        order: ordering of transformation
        e.g. 'TRS' -> 1)translate 2)rotate 3)scale
        e.g. 'RTS' -> 1)rotate 2)translate 3)scale
        '''

        # transforming the self.poin
        for opIdx, opType in enumerate(operTypes):
            
            if opType == 'T' and operVals[opIdx]!=(0,0):
                tx,ty = operVals[opIdx]
                self.point = self.point.translate(tx,ty)
                
            elif opType == 'R' and operVals[opIdx]!=0:
                theta = operVals[opIdx]
                ref = operRefs[opIdx]
                self.point = self.point.rotate(theta,ref)
                
            elif opType == 'S' and operVals[opIdx]!=(1,1):
                sx,sy = operVals[opIdx]
                ref = operRefs[opIdx]
                self.point = self.point.scale(sx,sy,ref)

        # updating self._traitTval after transforming the self.point
        self.update_tval(traits)

    ####################################
    def update_tval (self, traits):
        '''
        Node class
        '''
        self._traitTval = [traits[cIdx].IPE(self.point) for cIdx in self._traitIdx]
            

################################################################################
class HalfEdge:
    def __init__ (self,
                  selfIdx, twinIdx,
                  traitIdx, direction):
        '''
        '''
        self.attributes = {}
        self.selfIdx = selfIdx   # self-index (startNodeIdx, endNodeIdx, pathIdx)
        self.twinIdx = twinIdx   # index to the twin half-edge
        self.succIdx = None      # index to the successor half-edge

        # half edge Trait's attributes:
        self.traitIdx = traitIdx    # Index of the trait creating the edge
        self.direction = direction  # defines the direction of t-value (positive: t2>t1, negative: t1>t2)


    ####################################
    def get_tvals(self, traits, nodes):
        '''
        HalfEdge class

        usage: (sTVal, eTVal) = halfedge.get_tvals (traits, nodes)

        traits: traits stored in the arrangment (arrangment.traits)
        nodes: nodes of the main graph of the arrangment (arrangment.graph.node)

        sTVal: t-value of the trait corresponding to the starting node of the half-edge
        eTVal: t-value of the trait corresponding to the ending node of the half-edge

        trait.DPE(TVal) = node.point
        '''
        (s,e,k) = self.selfIdx

        sTVal = nodes[s]['obj']._traitTval[nodes[s]['obj']._traitIdx.index(self.traitIdx)]
        eTVal = nodes[e]['obj']._traitTval[nodes[e]['obj']._traitIdx.index(self.traitIdx)]

        # for the case of circles and the half-edge that crosses the theta=0=2pi
        if (self.direction=='positive') and not(sTVal < eTVal):
            eTVal += 2*np.pi
        if (self.direction=='negative') and not(sTVal > eTVal):
            sTVal += 2*np.pi

        return (sTVal, eTVal)
    

################################################################################
class Face:
    def __init__(self, halfEdgeList, path):
        '''
        '''
        self.halfEdges = halfEdgeList # a list of half-edges [(s,e,k), ...]
        self.path = path              # mpl.path
        self.holes = ()               # tuple of faces
        self.attributes = {}

    
    ####################################
    def get_all_halfEdges_Idx(self):
        '''
        Face class
        '''
        return [heIdx
                for hole in self.holes
                for heIdx in hole.halfEdges ]  + self.halfEdges
        
    ####################################
    def update_path(self, graph, traits):
        '''
        Face class
        '''
        self.path = edgeList_2_mplPath (self.halfEdges, graph, traits)

        for hole in self.holes:
            hole.path = edgeList_2_mplPath (hole.halfEdges, graph, traits)

    ####################################
    def get_area(self, considerHoles=True):
        '''
        Face class

        Be aware that path.to_polygons() is an approximation of the face,
        if it contains traits, consequently the area would be approximated

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

    ####################################
    def is_point_inside(self, point):
        '''
        Face class
        '''
        if self.path.contains_point( (point.x,point.y) ):
            for hole in self.holes:
                if hole.path.contains_point( (point.x,point.y) ):
                    return False
            return True        
        return False

    ####################################
    def punch_hole(self, holeFace):
        '''
        Face class

        although the path of the hole is suffiecent to detect inclusion of points,
        yet, holes are stored as faces, because the face instance of the hole is 
        required to able to traverse the boundary (half-edges) of the punched face.
        '''
       
        holes = list(self.holes)
        
        holeFace.holes = () # hole of hole is irrelevant
        holes.append(holeFace)

        # TODO:
        # add the list of halfEdges of the hole to the face?        

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

    ####################################
    def get_punched_path(self):
        '''
        Face class

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
    def __init__ (self, graph, traits, faces, superFaceIdx=None):
        '''
        '''
        self.graph = graph
        self.traits = traits

        if superFaceIdx is not None:
            f = list(faces)
            self.superFace = f.pop(superFaceIdx)
            self.faces = tuple(f)
        else:
            self.superFace = None
            self.faces = faces

    ####################################
    def find_face(self, point):
        '''
        Decomposition class
        '''
        for idx,face in enumerate(self.faces):
            if face.is_point_inside(point):
                return idx
        return None

    ####################################
    def find_mutual_halfEdges(self, f1Idx, f2Idx):
        '''
        Decomposition class
        '''

        mutualsIdx = []
                
        for (s,e,k) in self.faces[f1Idx].get_all_halfEdges_Idx():
            (ts,te,tk) = self.graph[s][e][k]['obj'].twinIdx
            if (ts,te,tk) in self.faces[f2Idx].get_all_halfEdges_Idx():
                mutualsIdx.append( (s,e,k) )
                mutualsIdx.append( (ts,te,tk) )

        return mutualsIdx


    ####################################
    def find_neighbours(self, faceIdx):
        '''
        Decomposition class
        '''
        
        twinsIdx = [ self.graph[s][e][k]['obj'].twinIdx
                     for (s,e,k) in self.faces[faceIdx].get_all_halfEdges_Idx() ]
        
        # picking the face which have any of the twins as their boundary
        # or the baoundary of their holes
        neighbours = []
        for fIdx,face in enumerate(self.faces):
            boundary = face.get_all_halfEdges_Idx()

            if any([ twin in boundary  for twin in twinsIdx]):
                neighbours.append(fIdx)

        # identical:
        # neighbours = [ fIdx
        #                for fIdx,face in enumerate(self.faces)
        #                if any([ twin in face.get_all_halfEdges_Idx()
        #                         for twin in twinsIdx]) ]

        # rejecting the face itself if it is included in the neighbours
        if faceIdx in neighbours:
            neighbours.pop( neighbours.index(faceIdx) )
    
        return neighbours

    ####################################        
    def get_extents(self):
        '''
        Decomposition class
        '''
        bboxes = [face.path.get_extents() for face in self.faces]
        return matplotlib.transforms.BboxBase.union(bboxes)

    ####################################
    def does_intersect(self, other):
        '''
        Decomposition class

        checks only the intersection of the boundaries
        "other" could be: Face, Decomposition, Arrangement
        '''

        assert self.superFace

        if isinstance(other, Face):
            otherPath = other.path

        elif isinstance(other, Decomposition):
            assert other.superFace
            otherPath = other.superFace.path

        elif isinstance(other, Arrangement):
            assert other.decomposition.superFace
            otherPath = other.decomposition.superFace.path

        return self.superFace.path.intersects_path(otherPath,filled=False)

    ####################################
    def does_overlap(self, other):
        '''
        Decomposition class

        checks overlapping (and enclosure) of two regions
        "other" could be: Face, Decomposition, Arrangement
        '''

        assert self.superFace

        if isinstance(other, Face):
            otherPath = other.path

        elif isinstance(other, Decomposition):
            assert other.superFace
            otherPath = other.superFace.path

        elif isinstance(other, Arrangement):
            assert other.decomposition.superFace
            otherPath = other.decomposition.superFace.path

        return self.superFace.path.intersects_path(otherPath,filled=True)

    ####################################
    def does_enclose(self, other):
        '''
        Decomposition class

        checks if self encloses the other without boundary intersection
        "other" could only be on the Decomposition type
        '''
        assert self.superFace and other.superFace

        if self.does_overlap(other) and not(self.does_intersect(other)):
            sampleNodeIdx = other.graph.nodes()[0]
            samplePoint = other.graph.node[sampleNodeIdx]['obj'].point
            fIdx = self.find_face ( samplePoint )
            if fIdx != None:
                return True
        return False


    ####################################
    def update_face_path(self):
        '''
        Decomposition class
        '''
        for face in self.faces:
            face.update_path(self.graph, self.traits)

        if self.superFace != None:
            self.superFace.update_path(self.graph, self.traits)

        
################################################################################
############################################################## Arrangement class
####################################################### aggregates from networkx
################################################################################
class Arrangement:
    ############################################################################
    def __init__ (self, traits , config):
        '''
        traits are aggregated instances of sympy's geometric module
        (e.g. LineModified, CircleModified, ...)

        multiProcessing=0 -> no multi-processing
        multiProcessing=n -> n: number of processes
        
        self._end_point: adding end points of Ray, Segment and Arc as nodes
        '''
        self._multi_processing = config['multi_processing'] if ('multi_processing' in config.keys()) else 0
        self._end_point = config['end_point'] if ('end_point' in config.keys()) else False
        self._timing = config['timing'] if ('timing' in config.keys()) else False

        ########## reject duplicated traits and store internally
        self.traits = []
        self._store_traits(traits)

        ########## construct the base graph
        # TODO: I know I need a directional-multi-graph, explain why
        tic = time.time()
        self.graph = nx.MultiDiGraph()
        if self._timing: print( 'Graph construction time:\t', time.time() - tic )

        #### STAGE A: construct nodes
        tic = time.time()
        self._construct_nodes()
        if self._timing: print( 'nodes construction time:\t', time.time() - tic )

        #### STAGE B: construct edges
        tic = time.time()
        self._construct_edges()
        if self._timing: print( 'edges construction time:\t', time.time() - tic )

        ########## decomposition
        tic = time.time()
        self._decompose()
        if self._timing: print( 'decomposition construction time:\t', time.time() - tic )



    ############################################################################
    def get_prime_graph(self):
        '''
        Arrangement class

        arrang.graph is the main graph. It contains all sub-graphs (disconnected).
        arrang.graph -> prime

        why wouldn't this work?
        prime = arrang.graph.to_undirected()
        because the correspondance between indices of the edges of the original graph
        and the prime graph is missing, hence we don't know which edge in prime
        corresponds to which face in the arrang.decomposition.faces
        '''

        prime = nx.MultiGraph()
        # adding edges will creat the nodes, so why do we add nodes first? redundant?
        # yes, redundant. But if there is a non-connected node in the original graph
        # it will be missed through adding edge process.
        all_nodes_idx = [ [ idx, {} ] for idx in self.graph.nodes() ] 
        prime.add_nodes_from( all_nodes_idx ) 
        

        all_halfedges_idx = [halfEdgeIdx for halfEdgeIdx in self.graph.edges(keys=True)]
        while len(all_halfedges_idx) != 0:
            (s,e,k) = all_halfedges_idx[-1]
            (ts,te,tk) = self.graph[s][e][k]['obj'].twinIdx
            
            edge = ( s, e, {'corresponding_halfedges_idx': [(s,e,k), (ts,te,tk)]} )
            prime.add_edges_from([edge])
            
            all_halfedges_idx.pop( all_halfedges_idx.index((s,e,k)) )
            all_halfedges_idx.pop( all_halfedges_idx.index((ts,te,tk)) )

        return prime


    ############################################################################
    def get_dual_graph(self):
        '''
        Arrangement class\
        
        arrang.decomposition is the main decomposition. I contains all the faces.
        arrang.decomposition -> dual
        '''
        
        dual = nx.MultiGraph()

        # per each face, a node is created in the dual graph
        # nodes = [ [fIdx, {'face':face}] for fIdx,face in enumerate(self.decomposition.faces)]
        nodes = [ [fIdx, {}] for fIdx,face in enumerate(self.decomposition.faces)]
        dual.add_nodes_from( nodes )

        # for every pair of faces an edge is added to the dual graph if they are neighbours
        for (f1Idx,f2Idx) in itertools.combinations( range(len(self.decomposition.faces) ), 2):
            mutualHalfEdges = self.decomposition.find_mutual_halfEdges(f1Idx, f2Idx)
            if len(mutualHalfEdges) !=0 :
                # note that even if two faces share more than one pair of twin half-edges,
                # still there will be one connecting edge between the two.
                dual.add_edges_from( [ (f1Idx,f2Idx, {}) ] )
                # dual.add_edges_from( [ (f1Idx,f2Idx, {'mutualHalfEdges': mutualHalfEdges}) ] )

        return dual
    


    ############################################################################
    def merge_faces(self, faceIdx=[]):
        '''
        Arrangement class
        '''

        # to reject duplication
        faceIdx = list(set(faceIdx))
        
        halfEdge2Remove = []

        for (f1Idx,f2Idx) in itertools.combinations(faceIdx, 2):
            # f1 = self.decomposition.faces[f1Idx]
            # f2 = self.decomposition.faces[f2Idx]

            halfEdge2Remove += self.decomposition.find_mutual_halfEdges(f1Idx, f2Idx)

        # remove all halfEdges, that separate the iput faces, from the graph
        self.graph.remove_edges_from( list(set(halfEdge2Remove)) )

        # check if any node is loose and not connected
        nodeDegrees = self.graph.degree(self.graph.nodes())
        self.graph.remove_nodes_from( [k
                                       for k in nodeDegrees.keys()
                                       if nodeDegrees[k] == 0 ] )
        
        # recompute the decomposition
        self._decompose()

    ############################################################################
    def _decompose(self):
        '''
        Arrangement class
        '''

        #### STAGE 0: split the base graph into connected subgraphs
        subGraphs = list(nx.connected_component_subgraphs(self.graph.to_undirected()))
        subGraphs = [sg.to_directed() for sg in subGraphs]
        
        #### STAGE A: decomposition of each subgraph and merging
        subDecompositions = []
        for subGraphIdx, sg in enumerate( subGraphs ):
            faces = self._decompose_graph(sg)
            if len(faces) == 0:
                # we need to check if self._decompose_graph(sg) returns anything
                # for instance, if two line intersect only with each other,
                # there will be a subgraph of one node, with no edge or face
                # therefore, no decomposition shall be stored
                superFaceIdx = None

            # find the superFace of the decomposition
            else:
                if len(faces) == 2:
                    # if only there are two faces, they will have the same area size
                    # so we look into the side attribute of half-edges of each face
                    # the one with negative side is selected as superFace

                    if len(faces[0].halfEdges) == 1:
                        # this is the case of a single circle
                        (s,e,k) = faces[0].halfEdges[0]
                        direction = self.graph[s][e][k]['obj'].direction
                        superFaceIdx = 0 if direction=='negative' else 1

                    else:
                        # TODO: I have no idea whether this is correct! check!
                        # hypothesis:
                        # when a arrangement returns only two face, of which one is superFace
                        # the sum of the internal angles of superFace is always bigger than 
                        # the corresponding value of the inner face.
                        angleSum = [0,0]
                        for fIdx, face in enumerate(faces):
                            for (s,e,k) in face.halfEdges:                            
                                # ta: twin's departure angle
                                (ts,te,tk) = self.graph[s][e][k]['obj'].twinIdx
                                twin = self.graph[ts][te][tk]['obj']
                                trait = self.traits[twin.traitIdx]
                                ta = trait.tangentAngle( self.graph.node[ts]['obj'].point,
                                                         twin.direction)
                                
                                # sa: successor's departure angle
                                (ss,se,sk) = self.graph[s][e][k]['obj'].succIdx
                                succ = self.graph[ss][se][sk]['obj']
                                trait = self.traits[succ.traitIdx]
                                sa = trait.tangentAngle( self.graph.node[ss]['obj'].point,
                                                         succ.direction)
                                
                                # sa, ta in [0,2pi]
                                # and we want da to be from ta to sa in ccw direction
                                da = ta - sa if ta>sa else (ta+2*np.pi) - sa
                                angleSum[fIdx] += da
                                
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
                    
            subDecompositions.append( Decomposition(sg, self.traits,
                                                    faces, superFaceIdx) )

        #### STAGE B: find holes of subDecompositions and punch holes
        self._find_punch_holes(subDecompositions)
        self._subDecompositions = subDecompositions

        #### STAGE C:
        # decomposition <- all faces and superFaces together
        allFaces = ()
        superfaces = []
        for sd in self._subDecompositions:
            if len(sd.faces)>0:  # sd could be empty
                allFaces += sd.faces
                superfaces += [sd.superFace]
                # todo:
                # a) reject superfaces inside bigger superfaces
                # b) should superfaces be a list of all superFaces, or should we combine them?
                # class.method make_compound_path(*args) Make a compound path from a list of Path objects.

        self.decomposition = Decomposition(self.graph, self.traits,
                                           allFaces, superFaceIdx=None)


    ############################################################################
    def _find_punch_holes(self, subDecompositions):

        '''
        Arrangement class
        '''        
        for (idx1,idx2) in itertools.permutations(range(len(subDecompositions)), 2):
            sd1 = subDecompositions[idx1]
            sd2 = subDecompositions[idx2]

            # sd1 or sd2 could be empty; faces =[] and superFace=None
            if len(sd1.faces)>0 and len(sd2.faces)>0 and sd1.does_enclose(sd2):
                sampleNodeIdx = sd2.graph.nodes()[0]
                samplePoint = sd2.graph.node[sampleNodeIdx]['obj'].point
                fIdx = sd1.find_face ( samplePoint )
                subDecompositions[idx1].faces[fIdx].punch_hole ( sd2.superFace )

        # for idx1,sd1 in enumerate(subDecompositions):
        #     for idx2,sd2 in enumerate(subDecompositions):
        #         if idx1 != idx2:
                    
        #             # sd1 or sd2 could be empty; faces =[] and superFace=None
        #             if len(sd1.faces)>0 and len(sd2.faces)>0 and sd1.does_enclose(sd2):
        #                     sampleNodeIdx = sd2.graph.nodes()[0]
        #                     samplePoint = sd2.graph.node[sampleNodeIdx]['obj'].point
        #                     fIdx = sd1.find_face ( samplePoint )
        #                     subDecompositions[idx1].faces[fIdx].punch_hole ( sd2.superFace )

    ############################################################################
    def _store_traits(self, traits):
        '''
        Arrangement class

        discard overlapping and invalid traits

        note that if two traits are ovelapping, the one with 
        higher index in the list "traits" will be rejected

        invalid traits are:    circles/arcs, where radius <= 0
        '''
        epsilon = np.spacing(10**10)

        for cIdx1 in range(len(traits)-1,-1,-1):
            obj1 = traits[cIdx1].obj
            obj1IsCirc = isinstance(traits[cIdx1], trts.CircleModified)
            obj1IsArc = isinstance(traits[cIdx1], trts.ArcModified)
            # note that a an arc both obj1IsCirc and obj1IsArc are True
           
            if obj1IsCirc and traits[cIdx1].obj.radius<=0:
                # rejecting circles (and arcs) with (radius <= 0)
                traits.pop(cIdx1)

            elif isinstance(traits[cIdx1].obj, sym.Point):
                # if two ends of a segment are collocated, it's a point
                traits.pop(cIdx1)

            else:
                # rejecting overlapping traits
                for cIdx2 in range(cIdx1):
                    obj2 = traits[cIdx2].obj
                    obj2IsArc = isinstance(traits[cIdx2], trts.ArcModified)
                    
                    if obj1.contains(obj2) or obj2.contains(obj1):
                        if obj1IsArc and obj2IsArc:
                            # TODO: here here check containment
                            # assuming all the angles are in [-pi,2pi]                            
                            # arc1t1 = traits[cIdx1].t1
                            # arc1t2 = traits[cIdx1].t2
                            # arc2t1 = traits[cIdx2].t1
                            # arc2t2 = traits[cIdx2].t2
                            # if t1dis < epsilon and t2dis < epsilon:
                            #     traits.pop(cIdx1)
                            #     break
                            pass
                        else:
                            # circles, lines, segments and rays are handled here
                            traits.pop(cIdx1)
                            break

        self.traits = traits

    ############################################################################
    def _construct_nodes(self):
        '''
        Arrangement class

        |STAGE A| of Graph construction: node construction
        first we need a list of all intersections,
        while we can retrieve informations about each intersection point,
        such as which traits are intersecting at that point and what
        are the corresponding t-value of each trait's parametric expression
        at that intersection point.
        these informations are important to construct the nodes:

        intersectionsFlat ( <- intersections )
        ipsTraitIdx : trait indices of each ips/node
        ipsTraitTVal : ips/node's tValue over each assigned trait

        # step 1: finding all intersections
        # step 2: reject an intersection if it is not a point
        # step 3: handling non-intersecting traits
        # step 4: flattening the intersections [list-of-lists-of-lists] -> [list of lists]
        # step 5: adding two virtual intersection points at the -oo and +oo
        # step 6: find indeces of traits corresponding to each intersection point
        # step 7: merge collocated intersection points
        # step 8: find the t-value of each trait at the intersection
        # step 9: creating nodes from >intersection points<

        intersections
        this variable is a 2d matrix (list of lists) where each element at
        intersections[row][col] is itself a list of intersection points between
        two traits self.traits[row] and self.traits[col].
        '''

        intersections = [] # 2D array storage of intersection points

        # the indices to all following 3 lists are the same, i.e. ips_idx
        intersectionsFlat = []   # 1D array storage of intersection points
        ipsTraitIdx = []         # ipsTraitIdx[i]: idx of traits on nodes[i]
        ipsTraitTVal = []        # t-value of each node at assigned traits


        ########################################
        # step 1: finding all intersections
        intersections = [ [ []
                            for col in range(len(self.traits)) ]
                          for row in range(len(self.traits)) ]

        if self._multi_processing: # with multi_processing
            traitsTuplesIdx = [ [row,col]
                                for row in range(len(self.traits))
                                for col in range(row) ]

            global traits
            traits = self.traits
            with ctx.closing(mp.Pool(processes=self._multi_processing)) as p:
                intersections_tmp = p.map( intersection_star, traitsTuplesIdx)
            del traits, p
            

            
            for (row,col),ips in zip (traitsTuplesIdx, intersections_tmp):
                intersections[row][col] = ips
                intersections[col][row] = ips
            del col,row, ips
                
        else:  # without multi_processing
            for row in range(len(self.traits)):
                for col in range(row):
                    obj1 = self.traits[row].obj
                    obj2 = self.traits[col].obj
                    ip_tmp = sym.intersection(obj1,obj2)
                    intersections[row][col] = ip_tmp
                    intersections[col][row] = ip_tmp
            del col, row, ip_tmp


        ######################################## TO DELETE    
        # self.all_intersection_points = intersections
        # print ('np.shape(intersections):', np.shape(intersections))
        ######################################## TO DELETE


        ########################################
        # step 2: reject an intersection if it is not a point
        for row in range(len(self.traits)):
            for col in range(row):
                ips = intersections[row][col]
                if len(ips)>0 and isinstance(ips[0], sym.Point):
                    pass
                else:
                    intersections[row][col] = []
                    intersections[col][row] = []
        del col,row, ips

        ########################################
        # step 3: handling non-intersecting Traits
        # It would be problematic if a Trait does not intersect with any other
        # traits. The problem is that the construcion of the nodes relies on
        # the detection of the intersection points. Also the construction of
        # edges relies on the intersection points lying on each Trait.
        # No edge would result from a Trait without any intersection.
        # Consequently that Traits, not be presented by any edges, would the
        # be missed through the decomposition method (which relies on the edges)
        # >> This won't happen for unbounded traits (e.g. infinit
        # lines), since the -oo and +oo are assigned to them as intersection
        # points, however, this is a potential risk for bounded traits (e.g.
        # circles).        
        # >> To avoid this problem we can assign an arbitrary point on the
        # level-trait of the Traits, as a self-intersection point. This
        # self-intersection point won't mess the later stages even in case of
        # traits that already intesect with other traits. Hencefore for
        # the sake of simplicity, we add a self-intersection point to all
        # bounded traits.

        t = sym.Symbol('t')
        for row in range(len(self.traits)):
            traitIsCircle = isinstance(self.traits[row], trts.CircleModified)
            traitIsCircle = traitIsCircle and not(isinstance(self.traits[row], trts.ArcModified))
            if traitIsCircle:
                ips_n = np.sum( [ len(intersections[row][col])
                                  for col in range(len(self.traits)) ] )
                if ips_n==0:
                    p = self.traits[row].obj.arbitrary_point(t)
                    intersections[row][row] = [ p.subs([(t,0)]).evalf() ]

        ########################################
        # step x: adding end points of Ray, Segment and Arc as nodes
        t = sym.Symbol('t')
        if self._end_point == True:

            for row in range(len(self.traits)):
                # trait is arc
                if isinstance(self.traits[row], trts.ArcModified):
                    p = self.traits[row].obj.arbitrary_point(t)
                    intersections[row][row] += [ p.subs([(t, self.traits[row].t1)]).evalf(),
                                                 p.subs([(t, self.traits[row].t2)]).evalf()]
                # trait is segment
                elif isinstance(self.traits[row], trts.SegmentModified):
                    intersections[row][row] += [ self.traits[row].obj.p1,
                                                 self.traits[row].obj.p2]

                # trait is Ray
                elif isinstance(self.traits[row], trts.RayModified):
                    intersections[row][row] += [ self.traits[row].obj.p1 ]
            

        ########################################
        # step 4: flattening the intersections list-of-lists-of-lists
        intersectionsFlat = [p
                             for row in range(len(self.traits))
                             for col in range(row+1) # for self-intersection
                             for p in intersections[row][col] ]

        ########################################
        # step 6: find indeces of traits corresponding to each intersection point
        ipsTraitIdx = [list(set([row,col]))
                       for row in range(len(self.traits))
                       for col in range(row+1) # for self-intersection 
                       for p in intersections[row][col] ]

        ########################################
        # step 7: merge collocated intersection points
        # duplicate: resulted of same traits intersection
        # collocated: resulted of different traits intersection

        # ips = np.array([[1,4],
        #                 [2,5],
        #                 [3,6]])
        
        ips_ = np.array(intersectionsFlat, dtype=np.float)
        
        xh = np.repeat( [ips_[:,0]], ips_.shape[0], axis=0)
        xv = np.repeat( [ips_[:,0]], ips_.shape[0], axis=0).T
        dx = xh - xv
        
        yh = np.repeat( [ips_[:,1]], ips_.shape[0], axis=0)
        yv = np.repeat( [ips_[:,1]], ips_.shape[0], axis=0).T
        dy = yh - yv

        distances = np.sqrt( dx**2 + dy**2)
        
        for idx1 in range(len(intersectionsFlat)-1,-1,-1):
            for idx2 in range(idx1):
                if distances[idx1][idx2] < np.spacing(10**10): # == 0:
                    intersectionsFlat.pop(idx1)
                    s1 = set(ipsTraitIdx[idx1])
                    s2 = set(ipsTraitIdx[idx2])
                    ipsTraitIdx[idx2] = list(s1.union(s2)) 
                    ipsTraitIdx.pop(idx1)
                    # print ('I\'m poping:' , idx1, '\T distance was: ', distances[idx1][idx2])
                    break

        assert len(intersectionsFlat) == len(ipsTraitIdx)

        ########################################
        # step 9: creating nodes from >intersection points<
        '''
        pIdx: intersection point's index
        cIdx: intersecting traits' indices
        tVal: intersecting traits' t-value at the intersection point
        '''
        nodes = [ [ pIdx, {'obj': Node(pIdx, intersectionsFlat[pIdx], ipsTraitIdx[pIdx], self.traits)} ]
                  for pIdx in range(len(intersectionsFlat)) ]
        
        ########################################
        # adding nodes to the graph
        self.graph.add_nodes_from( nodes )
        assert len(self.graph.nodes()) == len(intersectionsFlat)


    ############################################################################
    def _construct_edges(self):
        '''
        Arrangement class


        |STAGE B| of Graph construction: edge construction
        to create edges, we need to list all the nodes
        located on each Trait, along with the t-value of the Trait
        at each node to sort nodes over the trait and segment the
        trait into edges according to the sorted nodes
        '''

        ########################################
        # step 1_a: find intersection points of traits
        # indeces of intersection points corresponding to each traits
        traitIpsIdx = [[] for i in range(len(self.traits))]
        traitIpsTVal = [[] for i in range(len(self.traits))]

        for nodeIdx in self.graph.nodes():
            for (tVal,cIdx) in zip(self.graph.node[nodeIdx]['obj']._traitTval,
                                   self.graph.node[nodeIdx]['obj']._traitIdx) :
                traitIpsIdx[cIdx].append(nodeIdx)
                traitIpsTVal[cIdx].append(tVal)

        # step 1_b: sort intersection points over traits
        # sorting iptersection points, according to corresponding tVal
        for cIdx in range(len(self.traits)):
            tmp = sorted(zip( traitIpsTVal[cIdx], traitIpsIdx[cIdx] ))
            traitIpsIdx[cIdx] = [pIdx for (tVal,pIdx) in tmp]
            traitIpsTVal[cIdx].sort()


        # ########################################
        # step 3: half-edge construction
        for (cIdx,trait) in enumerate(self.traits):

            ipsIdx = traitIpsIdx[cIdx]
            tvals = traitIpsTVal[cIdx]

            # step a:
            # for each trait, create all edges (half-edges) located on it
            if isinstance(trait.obj, ( sym.Line, sym.Segment, sym.Ray) ):

                startIdxList = ipsIdx[:-1]
                startTValList = tvals[:-1]

                endIdxList = ipsIdx[1:]
                endTValList = tvals[1:]

            elif isinstance(trait, trts.ArcModified): # and isinstance(trait.obj, sym.Circ)

                # Important note: The order of elif matters...
                # isinstance(circle, trts.ArcModified) - > False
                # isinstance(circle, trts.CircleModified) - > True
                # isinstance(arc, trts.ArcModified) - > True
                # isinstance(arc, trts.CircleModified) - > True
                # this is why I first check the arc and then circle

                #TODO:  double-check
                startIdxList = ipsIdx[:-1]
                startTValList = tvals[:-1]

                endIdxList = ipsIdx[1:]
                endTValList = tvals[1:]
                
            elif isinstance(trait, trts.CircleModified): # and isinstance(trait.obj, sym.Circle)
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

                newPathKey1 = len(self.graph[sIdx][eIdx]) if eIdx in self.graph[sIdx].keys() else 0
                newPathKey2 = len(self.graph[eIdx][sIdx]) if sIdx in self.graph[eIdx].keys() else 0

                # in cases where sIdx==eIdx, twins will share the same key ==0
                # this will happen if there is only one node on a circle
                # also a non-intersecting circles with one dummy node
                # next line will take care of that only
                if sIdx==eIdx: newPathKey2 += 1
                
                idx1 = (sIdx, eIdx, newPathKey1)
                idx2 = (eIdx, sIdx, newPathKey2)

                # Halfedge(selfIdx, twinIdx, cIdx, side)

                # first half-edge
                direction = 'positive'                
                he1 = HalfEdge(idx1, idx2, cIdx, direction)
                e1 = ( sIdx, eIdx, {'obj':he1} )

                # second half-edge
                direction = 'negative'                
                he2 = HalfEdge(idx2, idx1, cIdx, direction)
                e2 = ( eIdx, sIdx, {'obj': he2} )

                self.graph.add_edges_from([e1, e2])

    # ############################################################################
    # def get_all_HalfEdge_indices (self, graph=None):
    #     '''
    #     Arrangement class
    #     replaced "arrang.get_all_HalfEdge_indices()" with "arrang.graph.edges(keys=True)"
        
    #     '''

    #     if graph==None: graph = self.graph

    #     allHalfEdgeIdx = [(sIdx, eIdx, k)
    #                       for sIdx in graph.nodes()
    #                       for eIdx in graph[sIdx].keys()
    #                       for k in graph[sIdx][eIdx].keys()]

    #     return allHalfEdgeIdx

    ############################################################################
    def _find_successor_HalfEdge(self, halfEdgeIdx, 
                                 allHalfEdgeIdx=None,
                                 direction='ccw_before'):
        '''
        Arrangement class
        '''

        # Note that in cases where there is a circle with one node on it,
        # the half-edge itself would be among candidates,
        # and it is important not to reject it from the candidates
        # otherwise the loop for find the face will never terminate!
        
        if allHalfEdgeIdx == None:
            allHalfEdgeIdx = self.graph.edges(keys=True) # self.get_all_HalfEdge_indices(self.graph) here here

        (start, end, k) = halfEdgeIdx

        # "candidateEdges" are those edges starting from the "end" node
        candidateEdges = [idx
                          for (idx, heIdx) in enumerate(allHalfEdgeIdx)
                          if (heIdx[0] == end)]# and openList[idx] ]

        # reject the twin half-edge of the current half-edge from the "candidateEdges"
        twinIdx = self.graph[start][end][k]['obj'].twinIdx
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
            # we have to let the path return by the twin
            # this will happen in cases of Ray, Segment and Arc
            
            # TODO:
            # this unforunately will result in a having faces with null area
            # if a subgraph contains no cycle (i.e. a tree)

            return allHalfEdgeIdx.index(twinIdx)

        else:
            # reference: the 1st and 2nd derivatives of the twin half-edge
            (tStart, tEnd, tk) = twinIdx
            refObj = self.graph[tStart][tEnd][tk]['obj']

            # sorting values of the reference (twin of the current half-edge)
            # 1stKey: alpha - 2ndkey: beta
            refObjTrait = self.traits[refObj.traitIdx]
            sPoint = self.graph.node[tStart]['obj'].point
            refAlpha = refObjTrait.tangentAngle(sPoint, refObj.direction)
            refBeta = refObjTrait.curvature(sPoint, refObj.direction)

            # sorting values: candidates
            canAlpha = []
            canBeta = []
            for candidateIdx in candidateEdges:
                (cStart, cEnd, ck) = allHalfEdgeIdx[candidateIdx]
                canObj = self.graph[cStart][cEnd][ck]['obj']
                canObjTrait = self.traits[canObj.traitIdx]
                canAlpha.append( canObjTrait.tangentAngle(sPoint, canObj.direction) )
                canBeta.append( canObjTrait.curvature(sPoint, canObj.direction) )

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

            return successorIdx


    ############################################################################
    def _decompose_graph(self, graph):
        '''
        Arrangement class

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
        allHalfEdgeIdx = graph.edges(keys=True)

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

                nextHalfEdgeIdx = self._find_successor_HalfEdge( face_tmp[0],
                                                                 allHalfEdgeIdx,
                                                                 direction='ccw_before')

                while face_tmp[0] != nextHalfEdgeIdx:#sNodeIdx == eNodeIdx:

                    # find the next half-edge in the trajectory
                    nextHalfEdgeIdx = self._find_successor_HalfEdge( face_tmp[-1],
                                                                     allHalfEdgeIdx,
                                                                     direction='ccw_before')

                    # update the face_tmp, if the next half-edge is open
                    if openList[nextHalfEdgeIdx]:
                        face_tmp.append( allHalfEdgeIdx[nextHalfEdgeIdx] )
                        eNodeIdx = allHalfEdgeIdx[nextHalfEdgeIdx][1]
                        openList[nextHalfEdgeIdx] = 0
                    else:
                        break                        
                        # to be implemented later - or not!
                        # >> this will happen if one of the nodes is in infinity,
                        # which means we closed them in openList before.
                        # >> for now: ignore the face_tmp value,
                        # because it is the face that contains infinity!
                        # and the openList is updated properly so far

                if sNodeIdx == eNodeIdx:
                    # with this condition we check if the face closed
                    # or the "while-loop" broke, reaching an infinity
                    # connected half edge.
                    faces.append(face_tmp)

                else:
                    pass

        ####### assign successor halfEdge Idx to each halfEdge:
        for edgeList in faces:
            for idx in range(len(edgeList)-1):
                (cs,ce,ck) = edgeList[idx] # current halfEdgeIdx
                (ss,se,sk) = edgeList[idx+1] # successor halfEdgeIdx
                self.graph[cs][ce][ck]['obj'].succIdx = (ss,se,sk)
            (cs,ce,ck) = edgeList[-1] # current halfEdgeIdx
            (ss,se,sk) = edgeList[0] # successor halfEdgeIdx
            self.graph[cs][ce][ck]['obj'].succIdx = (ss,se,sk)


        return tuple( Face( edgeList,
                            edgeList_2_mplPath(edgeList, self.graph, self.traits ) )
                      for edgeList in faces )

    ############################################################################
    def transform_sequence(self, operTypes, operVals, operRefs, subDecompositions=False):
        '''
        Arrangement class
        '''      
        # update arrangement.traits
        for trait in self.traits:
            trait.transform_sequence(operTypes, operVals, operRefs)
      
        # update arrangement.graph (nodes)
        for nIdx in self.graph.nodes():
            self.graph.node[nIdx]['obj'].transform_sequence(operTypes, operVals, operRefs, self.traits)
        
        # update arrangement.decomposition.faces
        self.decomposition.update_face_path()


        # update decomposition.faces
        if subDecompositions==True:
            for subDec in self.subDecompositions:
                for nIdx in subDec.graph.nodes():
                    subDec.graph.node[nIdx]['obj'].transform_sequence(operTypes, operVals, operRefs, self.traits)
                subDec.update_face_path()

    ############################################################################
    def save_faces_to_svg_path(self,  fileName, resolution=10.):
        ''' 
        Arrangement class
        '''
        pass #TODO(saesha)

    ############################################################################
    def add_new_traits(self, traits=[]):
        '''
        Arrangement class
        '''
        pass #TODO(saesha)
