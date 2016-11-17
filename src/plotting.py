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

import numpy as np
import sympy as sym
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.transforms
import matplotlib.image as mpimg

from cStringIO import StringIO


################################################################################
################################################################ plotting graphs
################################################################################


######################################## graph plot with matplotlib
def plot_graph_networkx(graph):
    f, axes = plt.subplots(1)
    nx.draw(graph)
    plt.show()

######################################## graph plot with pydot
def plot_graph_pydot(graph):
    # http://stackoverflow.com/questions/10379448/plotting-directed-graphs-in-python-in-a-way-that-show-all-edges-separately
    # http://stackoverflow.com/questions/1664861/how-to-create-an-image-from-a-string-in-python
    
    d = nx.to_pydot(graph) # d is a pydot graph object, dot options can be easily set
    png_str = d.create_png()
    sio = StringIO() # file-like string, appropriate for imread below
    sio.write(png_str)
    sio.seek(0)

    f, axes = plt.subplots(1)
    img = mpimg.imread(sio)
    imgplot = plt.imshow(img)
    plt.show()


######################################## multi graph plot with pydot
def plot_multiple_graphs_pydot(graphs):

    f, axes = plt.subplots(len(graphs))

    for i,g in enumerate(graphs):
        d = nx.to_pydot(g) # d is a pydot graph object, dot options can be easily set
        png_str = d.create_png()
        sio = StringIO() # file-like string, appropriate for imread below
        sio.write(png_str)
        sio.seek(0)

        img = mpimg.imread(sio)
        imgplot = axes[i].imshow(img)

        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.axis('equal')
    plt.tight_layout()
    plt.show()

######################################## graph plot with pygraphviz
def plot_graph_pygraphviz(graph):
    # http://stackoverflow.com/questions/14943439/how-to-draw-multigraph-in-networkx-using-matplotlib-or-graphviz

    # # write graph to a dot file
    # nx.drawing.nx_pydot.write_dot(arrang.graph,'multi.dot')
    # !neato -T png multi.dot > multi.png
    pass

# agraph = nx.to_agraph(arrang.graph)
################################################################################
########################################################## interactive functions
################################################################################

##################################### interactive functions on click
def onClick(event):
    xe, ye = event.xdata, event.ydata
    if xe and ye:
        # print 'clicked: ', xe, ye

        global arrang
        point = sym.Point(xe, ye)
        faceIdx = arrang.decomposition.find_face(point)
        print faceIdx

    else:
        pass
        # print 'clicked out of border'

##################################### interactive functions on move
def onMove(event):
    xe, ye = event.xdata, event.ydata

    global arrang
    global ax

    if xe and ye:
        # print 'moving: ',  xe, ye
        point = sym.Point(xe, ye)
        fIdx = arrang.decomposition.find_face(point)

        if fIdx is not None: # explicit, because fIdx could be 0
            plot_new_face_with_patch(ax, faceIdx=fIdx)

    else:
        pass
        # print 'moving out of border'


################################################################################
######################################################### plotting decomposition
################################################################################

################################### plotting edges
def plot_edges(axis, arrang,
               alp=0.2, col='b',
               halfEdgeIdx=None,
               withArrow=False,
               printLabels=False):

    if halfEdgeIdx==None:
        halfEdgeList = arrang.graph.edges(keys=True) # arrang.get_all_HalfEdge_indices()
        
    else:
        halfEdgeList = halfEdgeIdx


    for (start, end, k) in halfEdgeList:
        
        he_obj = arrang.graph[start][end][k]['obj']
        curve_obj = arrang.curves[he_obj.curveIdx].obj
        
        thei =  he_obj.twinIdx
        sTVal, eTVal = he_obj.get_tvals(arrang.curves, arrang.graph.node)

        if isinstance(curve_obj, (sym.Line, sym.Segment, sym.Ray) ):
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
                    axis.plot ([x,x+dx], [y,y+dy], col, alpha=alp)

                if printLabels:
                    if he_obj.direction == 'positive':
                        axis.text( x+(dx/2), y+(dy/2),# + np.sqrt(dx**2 + dy**2)/1.0, 
                                   'e#'+str(start)+'-'+str(end)+'-'+str(k),
                                   fontdict={'color':col,  'size': 10})
                    elif he_obj.direction == 'negative':
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


################################### plotting nodes
def plot_nodes (axis, arrang, nodes=None,
               alp = 0.5, col = 'k',
               printLabels = False):

    if nodes==None:  nodes = arrang.graph.nodes()
    points = [arrang.graph.node[idx]['obj'].point for idx in nodes]
        
    nx = [p.x for p in points]
    ny = [p.y for p in points]
    axis.plot (nx,ny, col+'o', alpha= alp)

    if printLabels:
        font = {'color':col, 'size': 10}
        for idx in arrang.graph.nodes():            
            axis.text(arrang.graph.node[idx]['obj'].point.x,
                      arrang.graph.node[idx]['obj'].point.y,
                      'n#'+str(idx))



######################################## plot decomposition, no faces 
def plot_decomposition(arrangement,
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
        cid_click = fig.canvas.mpl_connect('button_press_event', onClick)

    if interactive_onMove:
        cid_move = fig.canvas.mpl_connect('motion_notify_event', onMove)

    if plotEdges:
        plot_edges (ax, arrang, printLabels=printEdgeLabels)
    if plotNodes:
        plot_nodes (ax, arrang, nodes=None, printLabels=printNodeLabels)

    # set axes limit
    bb = arrang.decomposition.get_extents()
    ax.set_xlim(bb.x0-1, bb.x1+1)#, ax.set_xticks([])
    ax.set_ylim(bb.y0-1, bb.y1+1)#, ax.set_yticks([])

    plt.axis('equal')
    plt.tight_layout()
    plt.show()


######################################## plot decomposition, face-> patch 
def plot_decomposition_colored (arrang,
                                printNodeLabels=True,
                                printEdgeLabels=False,
                                fCol='b', eCol='r'):

    # with plt.xkcd():
    fig = plt.figure( figsize=(12, 12) )
    ax = fig.add_subplot(111)

    plot_edges (ax, arrang, printLabels=printEdgeLabels)
    plot_nodes (ax, arrang, nodes=None, printLabels=printNodeLabels)

    for face in arrang.decomposition.faces:
        patch = mpatches.PathPatch(face.get_punched_path(),
                                   facecolor=fCol, edgecolor=eCol, alpha=0.5)
        ax.add_patch(patch)

    # set axes limit
    bb = arrang.decomposition.get_extents()
    ax.set_xlim(bb.x0-1, bb.x1+1)#, ax.set_xticks([])
    ax.set_ylim(bb.y0-1, bb.y1+1)#, ax.set_yticks([])

    ax.axis('equal')
    plt.tight_layout()
    plt.show()





################################################################################
################################################## animatining face with patches
################################################################################


######################################## 
def animate_face_patches(arrangement, timeInterval=1000):

    fig = plt.figure( figsize=(12, 12) )
    ax = fig.add_subplot(111)

    global arrang
    global face_counter
    arrang = arrangement
    face_counter = -1

    # Create a new timer object. 1000 is default
    timer = fig.canvas.new_timer(interval=timeInterval)
    # tell the timer what function should be called.
    timer.add_callback(plot_new_face_with_patch, ax)
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


#########################################
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

    if False: plt.savefig('face #'+str(faceIdx)+'.png')


################################################################################
######################################################### animatining Half-edges
################################################################################


########################################
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


######################################### half-edge plots
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
        print 'something is wrong!'

    # ##########################################################################
    # ################################# drawing derivatives of the new haldfedge
    # if False:
    #     p1 = arrang.graph.node[start]['obj'].point
    #     px, py = p1.x.evalf() , p1.y.evalf()
    #     he_obj = arrang.graph[start][end][k]['obj']

    #     # Blue: 1st derivative - tangent to the curve
    #     dx,dy = he_obj.s1stDer
    #     axis.arrow(px,py, dx,dy,
    #                length_includes_head = True,
    #                head_width = 0.5, head_length = 1.,
    #                fc = 'b', ec = 'b') , 

    #     # Green: normal to the 1st derivative
    #     dx,dy = he_obj.s1stDer
    #     dxn,dyn = np.array( [dy,-dx] ) if he_obj.direction =='positive' else np.array( [-dy,dx] )
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
    bb = arrang.decomposition.get_extents()
    axis.set_xlim(bb.x0-1, bb.x1+1)#, ax.set_xticks([])
    axis.set_ylim(bb.y0-1, bb.y1+1)#, ax.set_yticks([])

    # drawing/applying changes to the figure
    axis.figure.canvas.draw()

    if False: plt.savefig('half_edge #'+str(halfEdge_counter)+'.png')
