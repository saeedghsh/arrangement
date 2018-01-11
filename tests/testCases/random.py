# elif name == 'Random':
#     ''' Random case'''
#     curves = []
#     nl = 4
#     X1 = np.random.random(nl)
#     Y1 = np.random.random(nl)
#     X2 = np.random.random(nl)
#     Y2 = np.random.random(nl)
#     curves += [ Line( args=(Point(x1,y1), Point(x2,y2)) )
#                for (x1,y1,x2,y2) in zip(X1,Y1,X2,Y2) ]

#     nc = 2
#     Xc = np.random.random(nc)
#     Yc = np.random.random(nc)
#     Rc = np.random.random(nc) + .75
#     curves += [ Circle( args=(Point(xc,yc), rc) )
#                for (xc,yc,rc) in zip(Xc,Yc,Rc) ]

#     number_of_nodes = 0
#     number_of_edges = 0
#     number_of_faces = 0
#     number_of_subGraphs = 0


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
