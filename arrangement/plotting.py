'''
Copyright (C) Saeed Gholami Shahbandi. All rights reserved.
Author: Saeed Gholami Shahbandi

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

import numpy as np
import sympy as sym

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from . import geometricTraits as trts
except:
    from arrangement import geometricTraits as trts

################################################################################
########################################################## visualizing traits
################################################################################

def plot_traits(axis, traits, clrs=None, alph=None):
    if clrs is None: clrs = {'cir':'b', 'arc':'b', 'lin':'r', 'seg':'g', 'ray':'g'}
    if alph is None: alph = {'cir': 1., 'arc': 1., 'lin': 1., 'seg': 1., 'ray': 1.}

    for idx, trait in enumerate(traits):

        # note: order of the conditions matter since arcModified is subclass of CircleModified
        if isinstance( trait, trts.ArcModified ):
            t1,t2 = trait.t1 , trait.t2
            tStep = max( [np.int(np.abs(t2-t1)*(180/np.pi))+1 ,2])
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
            raise( StandardError( 'trait n#', str(idx), 'unknown') )

    return axis


################################################################################
########################################################## interactive functions
################################################################################
def onClick(event):
    xe, ye = event.xdata, event.ydata
    if xe and ye:
        global arrang
        point = sym.Point(xe, ye)
        faceIdx = arrang.decomposition.find_face(point)
        print( faceIdx )

def onMove(event):
    xe, ye = event.xdata, event.ydata
    if xe and ye:
        global arrang
        global ax
        point = sym.Point(xe, ye)
        fIdx = arrang.decomposition.find_face(point)
        if fIdx is not None: # explicit, because fIdx could be 0
            plot_new_face_with_patch(ax, faceIdx=fIdx)

        
################################################################################
######################################################### plotting decomposition
################################################################################
def plot_edges(axis, arrang,
               alp=1., col='b',
               halfEdgeIdx=None,
               withArrow=False,
               printLabels=False):
    '''
    halfEdgeIdx
    a list of half-edge indices to plot.
    If None, all will be plotted.
    '''
    edge_plot_instances = []
    if halfEdgeIdx is None:
        halfEdgeList = arrang.graph.edges(keys=True)
    else:
        halfEdgeList = halfEdgeIdx

    for (start, end, k) in halfEdgeList:
        he_obj = arrang.graph[start][end][k]['obj']
        trait_obj = arrang.traits[he_obj.traitIdx].obj
        sTVal, eTVal = he_obj.get_tvals(arrang.traits, arrang.graph.node)

        if isinstance(trait_obj, (sym.Line, sym.Segment, sym.Ray) ):
            if sTVal!=sym.oo and sTVal!=-sym.oo and eTVal!=sym.oo and eTVal!=-sym.oo:
                p1 = arrang.graph.node[start]['obj'].point
                p2 = arrang.graph.node[end]['obj'].point
                x, y = p1.x.evalf() , p1.y.evalf()
                dx, dy = p2.x.evalf()-x, p2.y.evalf()-y

                if withArrow:
                    axis.arrow( np.float(x),np.float(y),
                                np.float(dx),np.float(dy),
                                length_includes_head = True, shape='right',
                                linewidth = 1, head_width = 0.1, head_length = 0.2,
                                fc = col, ec = col, alpha=alp)
                else:
                    edge_plot_instances += [axis.plot ([x,x+dx], [y,y+dy], col, alpha=alp)[0]]

                if printLabels:
                    if he_obj.direction == 'positive':
                        axis.text( x+(dx/2), y+(dy/2),# + np.sqrt(dx**2 + dy**2)/1.0,
                                   'e#'+str(start)+'-'+str(end)+'-'+str(k),
                                   fontdict={'color':col,  'size': 10})
                    elif he_obj.direction == 'negative':
                        axis.text( x+(dx/2), y+(dy/2),# - np.sqrt(dx**2 + dy**2)/1.0,
                                   'e#'+str(start)+'-'+str(end)+'-'+str(k),
                                   fontdict={'color':col,  'size': 10})

        elif isinstance(trait_obj, sym.Circle):
            tStep = max( [np.int(np.abs(eTVal-sTVal)*(180/np.pi))+1 ,2])            
            theta = np.linspace(np.float(sTVal), np.float(eTVal),
                                tStep, endpoint=True)
            xc, yc, rc = trait_obj.center.x , trait_obj.center.y , trait_obj.radius
            x = xc + rc * np.cos(theta)
            y = yc + rc * np.sin(theta)
            if not withArrow:
                edge_plot_instances += [axis.plot (x, y, col, alpha=alp)[0]]

            elif withArrow:
                axis.plot (x[:-1], y[:-1], col, alpha=alp)
                axis.arrow( x[-2], y[-2],
                            np.float(x[-1]-x[-2]) , np.float(y[-1]-y[-2]),
                            length_includes_head = True, shape='right',
                            linewidth = 1, head_width = 0.1, head_length = 0.2,
                            fc = col, ec = col, alpha=alp)

            if printLabels:
                xp = x[len(x)//2]
                yp = y[len(x)//2]
                if he_obj.direction == 'positive':

                    axis.text(xp + (xc-xp)/10. ,
                              yp + (yc-yp)/10. ,
                               'e#'+str(start)+'-'+str(end)+'-'+str(k),
                               fontdict={'color':col,  'size': 10})

                elif he_obj.direction == 'negative':
                    axis.text(xp - (xc-xp)/10. ,
                              yp - (yc-yp)/10. ,
                               'e#'+str(start)+'-'+str(end)+'-'+str(k),
                               fontdict={'color':col,  'size': 10})

    return edge_plot_instances


def plot_nodes (axis, arrang, nodes=None,
               alp = 1., col = 'r',
               printLabels = False):
    if nodes==None:  nodes = arrang.graph.nodes()
    points = [arrang.graph.node[idx]['obj'].point for idx in nodes]

    node_plot_instances = []
    nx = [p.x for p in points]
    ny = [p.y for p in points]
    node_plot_instances = axis.plot (nx,ny, col+'o', alpha= alp)

    if printLabels:
        for idx in arrang.graph.nodes():
            axis.text(arrang.graph.node[idx]['obj'].point.x,
                      arrang.graph.node[idx]['obj'].point.y,
                      'n#'+str(idx))

    return node_plot_instances


def plot_decomposition(arrangement,
                       invert_axis = [],
                       interactive_onClick=False,
                       interactive_onMove=False,
                       plotNodes=False, printNodeLabels=False,
                       plotEdges=True, printEdgeLabels=False):

    if interactive_onClick or interactive_onMove:
        global arrang
        global ax
    arrang = arrangement

    fig = plt.figure( figsize=(12, 12) )
    ax = fig.add_subplot(111)

    if interactive_onClick:
        fig.canvas.mpl_connect('button_press_event', onClick)

    if interactive_onMove:
        fig.canvas.mpl_connect('motion_notify_event', onMove)

    if plotEdges:
        plot_edges (ax, arrang, printLabels=printEdgeLabels)
    if plotNodes:
        plot_nodes (ax, arrang, nodes=None, printLabels=printNodeLabels)

    # set axes limit
    bb = arrang.decomposition.get_extents()
    ax.set_xlim(bb.x0-1, bb.x1+1)
    ax.set_ylim(bb.y0-1, bb.y1+1)

    if 'x' in invert_axis: ax.invert_xaxis()
    if 'y' in invert_axis: ax.invert_yaxis()

    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_decomposition_colored (arrang,
                                printNodeLabels=True,
                                printEdgeLabels=False,
                                printFaceLabels=False,
                                fCol='b', eCol='r'):
    fig = plt.figure( figsize=(12, 12) )
    ax = fig.add_subplot(111)

    plot_edges (ax, arrang, printLabels=printEdgeLabels)
    plot_nodes (ax, arrang, nodes=None, printLabels=printNodeLabels)

    colors = plt.cm.gist_ncar(np.linspace(0, 1, len(arrang.decomposition.faces)))
    for f_idx,face in enumerate(arrang.decomposition.faces):
        patch = mpatches.PathPatch(face.get_punched_path(),
                                   facecolor=colors[f_idx][0:3], #fCol
                                   edgecolor=None, #eCol
                                   alpha=0.5)
        ax.add_patch(patch)

    # set axes limit
    bb = arrang.decomposition.get_extents()
    ax.set_xlim(bb.x0-1, bb.x1+1)
    ax.set_ylim(bb.y0-1, bb.y1+1)

    ax.axis('equal')
    plt.tight_layout()
    plt.show()


################################################################################
################################################## animatining face with patches
################################################################################
def animate_face_patches(arrangement, timeInterval=1000, back_img=None):
    fig = plt.figure( figsize=(8, 8) )
    ax = fig.add_subplot(111)      

    global arrang
    global face_counter
    arrang = arrangement
    face_counter = -1

    # Create a new timer object. 1000 is default
    timer = fig.canvas.new_timer(interval= int(timeInterval))
    # tell the timer what function should be called.
    timer.add_callback(plot_new_face_with_patch, ax)
    # start the timer
    timer.start()
    # timer.stop()

    if back_img is not None:
        ax.imshow(back_img, cmap='gray', alpha=.7, interpolation='nearest', origin='lower')

    # set axes limit
    bb = arrangement.decomposition.get_extents()
    margin = 1 if back_img is None else 10
    ax.set_xlim(bb.x0-margin, bb.x1+margin)
    ax.set_ylim(bb.y0-margin, bb.y1+margin)

    ax.set_xticks([])
    ax.set_yticks([])

    # plot
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    timer.stop()


def plot_new_face_with_patch(axis, faceIdx=None):
    global arrang

    if faceIdx is None: # explicit, because faceIdx could be 0
        global face_counter
        face_counter = np.mod(face_counter+1, len(arrang.decomposition.faces))
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
    plot_edges (axis, arrang, alp=0.1)

    # drawing new face via path-patch
    face = arrang.decomposition.faces[ faceIdx ]
    patch = mpatches.PathPatch(face.get_punched_path(),
                               facecolor='b', edgecolor='r', alpha=0.5)
                               # facecolor='r', edgecolor='k', alpha=0.5)
    axis.add_patch(patch)

    # print the face's index
    axis.text(0, 0,
              'face #'+str(faceIdx),
              fontdict={'color':'m', 'size': 25})

    # set axes limit
    bb = arrang.decomposition.get_extents()
    axis.set_xlim(bb.x0-1, bb.x1+1)#, ax.set_xticks([])
    axis.set_ylim(bb.y0-1, bb.y1+1)#, ax.set_yticks([])

    # draw changes on the canvas
    axis.figure.canvas.draw()

    if [False,True][0]: plt.savefig('{:05d}'.format(faceIdx)+'.png')


################################################################################
######################################################### animatining Half-edges
################################################################################
def animate_halfEdges(arrangement, timeInterval = .1*1000):

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)

    global arrang
    global halfEdge_counter
    arrang = arrangement
    halfEdge_counter = -1

    # plotting nodes
    plot_nodes (ax, arrang, printLabels = True)

    # Create a new timer object. 1000 is default
    timer = fig.canvas.new_timer(interval=timeInterval)
    # tell the timer what function should be called.
    timer.add_callback(plot_new_halfEdge, ax)
    # start the timer
    timer.start()
    # timer.stop()

    # set axes limit
    bb = arrangement.decomposition.get_extents()
    ax.set_xlim(bb.x0-1, bb.x1+1)#, ax.set_xticks([])
    ax.set_ylim(bb.y0-1, bb.y1+1)#, ax.set_yticks([])

    # plot
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    timer.stop()


def plot_new_halfEdge(axis):

    global halfEdge_counter
    global arrang

    # updating the counter, and fetching the next half edge
    allHalfEdgeIdx = arrang.graph.edges(keys=True) # arrang.get_all_HalfEdge_indices()
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
    plot_edges (axis, arrang, alp=0.1)

    # drawing new haldfedge
    halfEdge_direction = arrang.graph[start][end][k]['obj'].direction
    if halfEdge_direction == 'positive':
        plot_edges(axis, arrang,
                   halfEdgeIdx= [(start,end,k)],
                   alp=0.9, col='g',
                   withArrow=True)
    elif halfEdge_direction == 'negative':
        plot_edges(axis, arrang,
                   halfEdgeIdx= [(start,end,k)],
                   alp=0.9, col='r',
                   withArrow=True)
    else:
        print( 'something is wrong!' )

    # print index of the half-edge
    axis.text(-8, -7,
              'half-edge #' + str(start)+'-'+str(end)+'-'+str(k),
              fontdict={'color':'m', 'size': 25})

    # set axes limit
    bb = arrang.decomposition.get_extents()
    axis.set_xlim(bb.x0-1, bb.x1+1)#, ax.set_xticks([])
    axis.set_ylim(bb.y0-1, bb.y1+1)#, ax.set_yticks([])

    # drawing/applying changes to the figure
    axis.figure.canvas.draw()

    if False: plt.savefig('half_edge #'+str(halfEdge_counter)+'.png')
