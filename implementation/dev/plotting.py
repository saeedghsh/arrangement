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

import numpy as np
import sympy as sym
import networkx as nx

import matplotlib.pyplot as plt

import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.transforms

################################################################################
########################################################## interactive functions
################################################################################
def onClick(event):
    xe, ye = event.xdata, event.ydata
    if xe and ye:
        # print 'clicked: ', xe, ye

        global subdiv
        point = sym.Point(xe, ye)
        faceIdx = subdiv.decomposition.find_face(point)
        print faceIdx

    else:
        pass
        # print 'clicked out of border'

########################################
def onMove(event):
    xe, ye = event.xdata, event.ydata

    global subdiv
    global ax

    if xe and ye:
        # print 'moving: ',  xe, ye
        point = sym.Point(xe, ye)
        fIdx = subdiv.decomposition.find_face(point)

        if fIdx is not None: # explicit, because fIdx could be 0
            plot_new_face_with_patch(ax, faceIdx=fIdx)

    else:
        pass
        # print 'moving out of border'


######################################## graph plot
def plot_graph(graph):
    f, axes = plt.subplots(1)
    nx.draw(graph)
    plt.show()


################################### plotting edges
def plot_edges(axis, subdiv,
               alp=0.2, col='b',
               halfEdgeIdx=None,
               withArrow=False,
               printLabels=False):

    if halfEdgeIdx==None:
        halfEdgeList = subdiv.get_all_HalfEdge_indices()
    else:
        halfEdgeList = halfEdgeIdx

    for (start, end, k) in halfEdgeList:
        
        obj = subdiv.MDG[start][end][k]['obj']
        cIdx = obj.cIdx

        thei =  obj.twinIdx

        sTVal = obj.sTVal
        s1stDer = obj.s1stDer
        s2ndDer = obj.s2ndDer
        eTVal = obj.eTVal
        e1stDer = obj.e1stDer
        e2ndDer = obj.e2ndDer

        if isinstance(subdiv.curves[cIdx].obj, sym.Line):
            if sTVal!=sym.oo and sTVal!=-sym.oo and eTVal!=sym.oo and eTVal!=-sym.oo:
                p1 = subdiv.intersectionsFlat[start] #subd.curves[cIdx].DPE(sTVal)
                p2 = subdiv.intersectionsFlat[end] #subd.curves[cIdx].DPE(eTVal)
                x, y = p1.x.evalf() , p1.y.evalf()
                dx, dy = p2.x.evalf()-x, p2.y.evalf()-y

                if withArrow:
                    axis.arrow( np.float(x),np.float(y),
                                np.float(dx),np.float(dy),
                                length_includes_head = True, shape='right',
                                linewidth = 1, head_width = 0.1, head_length = 0.2,
                                fc = col, ec = col, alpha=alp)
                else:
                    axis.plot ([x,x+dx], [y,y+dy], col, alpha=alp)

                if printLabels:
                    axis.text( x+(dx/2), y+(dy/2) ,
                               'e#'+str(start)+'-'+str(end),
                               fontdict={'color':col,  'size': 16})

        elif isinstance(subdiv.curves[cIdx].obj , sym.Circle):
            tStep = max( [np.float(np.abs(eTVal-sTVal)*(180/np.pi)) ,2])
            theta = np.linspace(np.float(sTVal), np.float(eTVal),
                                tStep, endpoint=True)
            circ = subdiv.curves[cIdx].obj
            xc, yc, rc = circ.center.x , circ.center.y , circ.radius
            x = xc + rc * np.cos(theta)
            y = yc + rc * np.sin(theta)

            if not withArrow:
                axis.plot (x, y, col, alpha=alp)

            elif withArrow:
                axis.plot (x[:-1], y[:-1], col, alpha=alp)
                axis.arrow( x[-2], y[-2],
                            np.float(x[-1]-x[-2]) , np.float(y[-1]-y[-2]),
                            length_includes_head = True, shape='right',
                            linewidth = 1, head_width = 0.1, head_length = 0.2,
                            fc = col, ec = col, alpha=alp)

            if printLabels:
                axis.text( (x[0]+x[-1])/2,
                           (y[0]+y[-1])/2,
                           'e#'+str(start)+'-'+str(end),
                           fontdict={'color':col,  'size': 16})

################################### plotting nodes
def plot_nodes (axis, subdiv, nodes=None,
               alp = 0.5, col = 'k',
               printLabels = False):

    if nodes==None:
        points = subdiv.intersectionsFlat
    else:
        points = [subdiv.intersectionsFlat[idx] for idx in nodes]
        
    nx = [p.x for p in points]
    ny = [p.y for p in points]
    axis.plot (nx,ny, col+'o', alpha= alp)

    if printLabels:
        font = {'color':col, 'size': 10}
        for idx in range(len(subdiv.intersectionsFlat)):
            
            if subdiv.intersectionsFlat[idx].x != sym.oo and subdiv.intersectionsFlat[idx].x != -sym.oo:
                axis.text(subdiv.intersectionsFlat[idx].x,
                          subdiv.intersectionsFlat[idx].y,
                          'n#'+str(idx))



######################################## plot decomposition, no faces 
def plot_decomposition(subdivision,
                       interactive_onClick=False,
                       interactive_onMove=False,
                       plotNodes=False, printNodeLabels=False,
                       plotEdges=True, printEdgeLabels=False):

    if interactive_onClick or interactive_onMove:
        global subdiv
        global ax
        subdiv = subdivision

    fig = plt.figure( figsize=(12, 12) )
    ax = fig.add_subplot(111)

    if interactive_onClick:
        cid_click = fig.canvas.mpl_connect('button_press_event', onClick)

    if interactive_onMove:
        cid_move = fig.canvas.mpl_connect('motion_notify_event', onMove)

    if plotEdges:
        plot_edges (ax, subdivision, printLabels=printEdgeLabels)
    if plotNodes:
        plot_nodes (ax, subdivision, nodes=None, printLabels=printNodeLabels)

    # set axes limit
    bb = subdiv.decomposition.get_extents()
    ax.set_xlim(bb.x0-1, bb.x1+1)#, ax.set_xticks([])
    ax.set_ylim(bb.y0-1, bb.y1+1)#, ax.set_yticks([])

    plt.axis('equal')
    plt.tight_layout()
    plt.show()


######################################## plot decomposition, face-> patch 
def plot_decomposition_colored (subdiv,
                                printNodeLabels=True,
                                printEdgeLabels=False,
                                fCol='b', eCol='r'):

    # with plt.xkcd():
    fig = plt.figure( figsize=(12, 12) )
    ax = fig.add_subplot(111)

    plot_edges (ax, subdiv, printLabels=printEdgeLabels)
    plot_nodes (ax, subdiv, nodes=None, printLabels=printNodeLabels)

    for face in subdiv.decomposition.faces:
        patch = mpatches.PathPatch(face.get_punched_path(),
                                   facecolor=fCol, edgecolor=eCol, alpha=0.5)
        ax.add_patch(patch)

    # set axes limit
    bb = subdiv.decomposition.get_extents()
    ax.set_xlim(bb.x0-1, bb.x1+1)#, ax.set_xticks([])
    ax.set_ylim(bb.y0-1, bb.y1+1)#, ax.set_yticks([])

    ax.axis('equal')
    plt.tight_layout()
    plt.show()


################################################################################
##################################################################### animations
################################################################################

######################################### face - patch
def plot_new_face_with_patch(axis, faceIdx=None):

    global subdiv
    
    if faceIdx is None: # explicit, because faceIdx could be 0
        global face_counter
        face_counter = np.mod(face_counter+1, len(subdiv.decomposition.faces))
        faceIdx = face_counter
  
    # removing [almost] all plot instances
    for ch in axis.get_children():
        if ch.__str__()[0:4] == 'Poly':
            ch.remove()
        if ch.__str__()[0:4] == 'Line':
            ch.remove()
        elif ch.__str__()[0:4] == 'Text' and ch.get_text()[0:4]=='face':
            ch.remove()

    # redrawing the base
    plot_edges (axis, subdiv, alp=0.1)

    # drawing new face via path-patch
    face = subdiv.decomposition.faces[ faceIdx ]
    patch = mpatches.PathPatch(face.get_punched_path(),
                               facecolor='b', edgecolor='r', alpha=0.5)
                               # facecolor='r', edgecolor='k', alpha=0.5)
    axis.add_patch(patch)

    # print the face's index
    axis.text(0, 0,
              'face #'+str(faceIdx),
              fontdict={'color':'m', 'size': 25})

    # set axes limit
    bb = subdiv.decomposition.get_extents()
    axis.set_xlim(bb.x0-1, bb.x1+1)#, ax.set_xticks([])
    axis.set_ylim(bb.y0-1, bb.y1+1)#, ax.set_yticks([])

    # draw changes on the canvas
    axis.figure.canvas.draw()

    if False: plt.savefig('face #'+str(faceIdx)+'.png')

########################################
def animate_face_patches(subdivision):

    fig = plt.figure( figsize=(12, 12) )
    ax = fig.add_subplot(111)

    global subdiv
    global face_counter
    subdiv = subdivision
    face_counter = -1

    # Create a new timer object. 1000 is default
    timer = fig.canvas.new_timer(interval=1*1000)
    # tell the timer what function should be called.
    timer.add_callback(plot_new_face_with_patch, ax)
    # start the timer
    timer.start()
    # timer.stop()

    # set axes limit
    bb = subdivision.decomposition.get_extents()
    ax.set_xlim(bb.x0-1, bb.x1+1)#, ax.set_xticks([])
    ax.set_ylim(bb.y0-1, bb.y1+1)#, ax.set_yticks([])
    
    # plot
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    timer.stop()



######################################### Animating
######################################### half-edge plots
def plot_new_halfEdge(axis):

    global halfEdge_counter
    global subdiv
    
    # updating the counter, and fetching the next half edge
    allHalfEdgeIdx = subdiv.get_all_HalfEdge_indices() 
    halfEdge_counter = np.mod( halfEdge_counter+1, len(allHalfEdgeIdx) )
    (start,end,k) = allHalfEdgeIdx[halfEdge_counter]

    # removing all plot instances 
    for ch in axis.get_children():
        if ch.__str__()[0:4] == 'Line':
            ch.remove()
        elif ch.__str__() == 'FancyArrow()':
            ch.remove()
        elif ch.__str__()[0:4] == 'Text' and ch.get_text()[0:9]=='half-edge':
            ch.remove()

    # redrawing the base
    plot_edges (axis, subdiv, alp=0.1)

    # drawing new haldfedge
    plot_edges(axis, subdiv,
               halfEdgeIdx= [(start,end,k)],
               alp=0.9, col='m',
               withArrow=True)

    # print the face's index    
    axis.text(-8, -7,
              'half-edge #' + str(start)+'-'+str(end)+'-'+str(k),
              fontdict={'color':'m', 'size': 25})

    # set axes limit
    bb = subdiv.decomposition.get_extents()
    axis.set_xlim(bb.x0-1, bb.x1+1)#, ax.set_xticks([])
    axis.set_ylim(bb.y0-1, bb.y1+1)#, ax.set_yticks([])

    # drawing/applying changes to the figure
    axis.figure.canvas.draw()

    if False: plt.savefig('half_edge #'+str(halfEdge_counter)+'.png')

########################################
def animate_halfEdges(subdivision):

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)

    global subdiv
    global halfEdge_counter
    subdiv = subdivision
    halfEdge_counter = -1

    # plotting nodes
    plot_nodes (ax, subdiv, printLabels = True)

    # Create a new timer object. 1000 is default
    timer = fig.canvas.new_timer(interval=.1*1000)
    # tell the timer what function should be called.
    timer.add_callback(plot_new_halfEdge, ax)
    # start the timer
    timer.start()
    # timer.stop()

    # set axes limit
    bb = subdivision.decomposition.get_extents()
    ax.set_xlim(bb.x0-1, bb.x1+1)#, ax.set_xticks([])
    ax.set_ylim(bb.y0-1, bb.y1+1)#, ax.set_yticks([])

    # plot
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    timer.stop()







################################################################################
########################################################################### dump
################################################################################


# #################### plotting derivatives verctors
# def plot_derivatives (axis, subdiv,
#                       plotDerivatives1st = True,
#                       derivAlpha1st = 0.7, derivColor1st = 'r',
#                       plotDerivatives2nd = True,
#                       derivAlpha2nd = 0.7, derivColor2nd = 'b',
#                       printLabels = False):

#     for (start, end, k) in subdiv.allHalfEdgeIdx:
#         ps = subdiv.intersectionsFlat[start]
#         pe = subdiv.intersectionsFlat[end]

#         # attr = subdiv.MDG[start][end][k]

#         # cIdx = attr['cIdx']
#         # sTVal = attr['sTVal']

#         # thei =  attr['twinIdx']
#         # assert thei[0]==end and thei[1]==start

#         # s1stDer = attr['s1stDer']
#         # s2ndDer = attr['s2ndDer']
#         # eTVal = attr['eTVal']
#         # e1stDer = attr['e1stDer']
#         # e2ndDer = attr['e2ndDer']

#         obj = subdiv.MDG[start][end][k]['obj']
#         cIdx = obj.cIdx
#         sTVal = obj.sTVal
#         thei =  obj.twinIdx
#         s1stDer = obj.s1stDer
#         s2ndDer = obj.s2ndDer
#         eTVal = obj.eTVal
#         e1stDer = obj.e1stDer
#         e2ndDer = obj.e2ndDer

#         if plotDerivatives1st:
#             # at starting point
#             norm = np.sqrt(s1stDer.dot(s1stDer)) # np.linalg.norm(s1stDer,order=1)
#             if norm > 0:
#                 norm = 1.
#                 axis.arrow( ps.x, ps.y,
#                             s1stDer[0]/norm, s1stDer[1]/norm,
#                             length_includes_head = True,# shape='right',
#                             linewidth = 1, head_width = 0.1, head_length = 0.2,
#                             fc = derivColor1st, ec = derivColor1st, alpha=derivAlpha1st )

#                 if printLabels:
#                     if ps.x != sym.oo and ps.x !=-sym.oo:
#                         axis.text(ps.x + s1stDer[0]/norm,
#                                   ps.y + s1stDer[1]/norm,
#                                   'f#'+str(cIdx),
#                                   fontdict={'color':derivColor1st, 'size': 16})

#         if plotDerivatives2nd:
#             # at starting point
#             norm = np.sqrt(s2ndDer.dot(s2ndDer))
#             if norm > 0:
#                 norm = 1.
#                 axis.arrow( ps.x, ps.y,
#                             s2ndDer[0]/norm, s2ndDer[1]/norm,
#                             length_includes_head = True,# shape='right',
#                             linewidth = 1, head_width = 0.1, head_length = 0.2,
#                             fc = derivColor2nd, ec = derivColor2nd, alpha=derivAlpha2nd)

#                 if printLabels:
#                     axis.text(ps.x + s2ndDer[0]/norm,
#                               ps.y + s2ndDer[1]/norm,
#                               'c#'+str(cIdx),
#                               fontdict={'color':derivColor2nd, 'size': 16})

#             else:
#                 axis.plot( ps.x, ps.y,
#                            derivColor2nd+'o', fillstyle='none',
#                            linewidth=1, markersize=12, alpha=derivAlpha2nd)

#                 if printLabels:
#                     if ps.x != sym.oo and ps.x !=-sym.oo:
#                         axis.text( ps.x, ps.y,
#                                    'c#' + str(cIdx),
#                                    fontdict={'color':derivColor2nd, 'size': 16})




# ################################### plotting faces
# def plotFaces_half_edge(axis, subdiv,
#                         faceIdx=None,
#                         col='m', alp=0.8):

#     ''' plot face by its half-edges'''

#     faces = subdiv.faces if faceIdx is None else [subdiv.faces[idx]
#                                                   for idx in faceIdx]

#     facePlots = []

#     for face in faces:
#         for edge in face:

#             (start, end, k) = edge
#             obj = subdiv.MDG[start][end][k]['obj']
#             cIdx = obj.cIdx
#             sTVal = obj.sTVal
#             eTVal = obj.eTVal

#             if isinstance(subdiv.curves[cIdx].obj, sym.Line):
#                 if sTVal!=sym.oo and sTVal!=-sym.oo and eTVal!=sym.oo and eTVal!=-sym.oo:
#                     p1 = subdiv.intersectionsFlat[start]#subd.curves[cIdx].DPE(sTVal)
#                     p2 = subdiv.intersectionsFlat[end]#subd.curves[cIdx].DPE(eTVal)
#                     x, y = p1.x.evalf() , p1.y.evalf()
#                     dx, dy = p2.x.evalf()-x, p2.y.evalf()-y

#                     facePlots.append( axis.plot ([x,x+dx], [y,y+dy], col, alpha=alp) )

#             elif isinstance(subdiv.curves[cIdx].obj, sym.Circle):
#                 tStep = max( [np.float(np.abs(eTVal-sTVal)*(180/np.pi)) ,2])
#                 theta = np.linspace(np.float(sTVal), np.float(eTVal),
#                                     tStep, endpoint=True)
#                 circ = subdiv.curves[cIdx].obj
#                 xc, yc, rc = circ.center.x , circ.center.y , circ.radius
#                 x = xc + rc * np.cos(theta)
#                 y = yc + rc * np.sin(theta)

#                 facePlots.append( axis.plot (x, y, col,alpha=alp) )

#     return facePlots



# ################### plotting decomposing functions
# def plotCurves(axis, curves,
#               alp=0.4, col='b',
#               printLabels=False):

#     for c in curves:
#         if isinstance(c.obj , sym.Line):
#             x = [c.obj.p1.x , c.obj.p2.x]
#             y = [c.obj.p1.y , c.obj.p2.y]

#         elif isinstance(c.obj , sym.Circle):
#             x = c.obj.center.x + c.obj.radius * np.cos(np.linspace(0,2*np.pi,50, endpoint=True))
#             y = c.obj.center.y + c.obj.radius * np.sin(np.linspace(0,2*np.pi,50, endpoint=True))

#         axis.plot (x, y, col+'-', alpha=alp)

#         if printLabels:
#             pass

