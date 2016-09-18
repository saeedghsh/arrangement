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
        
        he_obj = subdiv.MDG[start][end][k]['obj']
        curve_obj = subdiv.curves[he_obj.cIdx].obj
        
        cIdx = he_obj.cIdx

        thei =  he_obj.twinIdx

        sTVal = he_obj.sTVal
        eTVal = he_obj.eTVal

        if isinstance(curve_obj, (sym.Line, sym.Segment, sym.Ray) ):
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
                    if he_obj.side == 'positive':
                        axis.text( x+(dx/2), y+(dy/2),# + np.sqrt(dx**2 + dy**2)/1.0, 
                                   'e#'+str(start)+'-'+str(end)+'-'+str(k),
                                   fontdict={'color':col,  'size': 10})
                    elif he_obj.side == 'negative':
                        axis.text( x+(dx/2), y+(dy/2),# - np.sqrt(dx**2 + dy**2)/1.0,
                                   'e#'+str(start)+'-'+str(end)+'-'+str(k),
                                   fontdict={'color':col,  'size': 10})

        elif isinstance(curve_obj, sym.Circle):
            tStep = max( [np.float(np.abs(eTVal-sTVal)*(180/np.pi)) ,2])
            theta = np.linspace(np.float(sTVal), np.float(eTVal),
                                tStep, endpoint=True)
            xc, yc, rc = curve_obj.center.x , curve_obj.center.y , curve_obj.radius
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
                xp = x[len(x)/2]
                yp = y[len(x)/2]
                if he_obj.side == 'positive':

                    axis.text(xp + (xc-xp)/10. ,
                              yp + (yc-yp)/10. ,
                               'e#'+str(start)+'-'+str(end)+'-'+str(k),
                               fontdict={'color':col,  'size': 10})
                    
                elif he_obj.side == 'negative':
                    axis.text(xp - (xc-xp)/10. ,
                              yp - (yc-yp)/10. ,
                               'e#'+str(start)+'-'+str(end)+'-'+str(k),
                               fontdict={'color':col,  'size': 10})








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
        plot_edges (ax, subdiv, printLabels=printEdgeLabels)
    if plotNodes:
        plot_nodes (ax, subdiv, nodes=None, printLabels=printNodeLabels)

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
def animate_face_patches(subdivision, timeInterval=1000):

    fig = plt.figure( figsize=(12, 12) )
    ax = fig.add_subplot(111)

    global subdiv
    global face_counter
    subdiv = subdivision
    face_counter = -1

    # Create a new timer object. 1000 is default
    timer = fig.canvas.new_timer(interval=timeInterval)
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
    halfEdge_side = subdiv.MDG[start][end][k]['obj'].side
    if halfEdge_side == 'positive':
        plot_edges(axis, subdiv,
                   halfEdgeIdx= [(start,end,k)],
                   alp=0.9, col='g',
                   withArrow=True)
    elif halfEdge_side == 'negative':
        plot_edges(axis, subdiv,
                   halfEdgeIdx= [(start,end,k)],
                   alp=0.9, col='r',
                   withArrow=True)
    else:
        print 'something is wrong!'

    # ##########################################################################
    # ################################# drawing derivatives of the new haldfedge
    # if False:
    #     p1 = subdiv.intersectionsFlat[start]
    #     px, py = p1.x.evalf() , p1.y.evalf()
    #     he_obj = subdiv.MDG[start][end][k]['obj']

    #     # Blue: 1st derivative - tangent to the curve
    #     dx,dy = he_obj.s1stDer
    #     axis.arrow(px,py, dx,dy,
    #                length_includes_head = True,
    #                head_width = 0.5, head_length = 1.,
    #                fc = 'b', ec = 'b') , 

    #     # Green: normal to the 1st derivative
    #     dx,dy = he_obj.s1stDer
    #     dxn,dyn = np.array( [dy,-dx] ) if he_obj.side =='positive' else np.array( [-dy,dx] )
    #     axis.arrow(px,py, dxn,dyn,
    #                length_includes_head = True,
    #                head_width = 0.5, head_length = 1.,
    #                fc = 'g', ec = 'g')

    #     # Red: 2ns derivative
    #     dx,dy = he_obj.s2ndDer
    #     axis.arrow(px,py, dx,dy,
    #                length_includes_head = True,
    #                head_width = 0.5, head_length = 1.,
    #                fc = 'r', ec = 'r')
    # ##########################################################################

    # print index of the half-edge
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
def animate_halfEdges(subdivision, timeInterval = .1*1000):

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)

    global subdiv
    global halfEdge_counter
    subdiv = subdivision
    halfEdge_counter = -1

    # plotting nodes
    plot_nodes (ax, subdiv, printLabels = True)

    # Create a new timer object. 1000 is default
    timer = fig.canvas.new_timer(interval=timeInterval)
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

