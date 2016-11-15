>>> Dual graph
topological adjacency graph -> subdiv.graph
geometric adjacency graph -> subdiv.graph + half-edges
topological connectivity graph -> subdiv.decomposition.faces
geometric connectivity graph -> ????

>>> Insert new curve:
Intersect the new curve with all other curves, break it into half-edges, take a point on each half-edge and find the face it belongs to. Split the face. The problem is that my data structure of the arrangement is not flat, even worse, it is nested. I'm talking about holes.

>>> Speed up intersection process.

>>> [subdivision] [CGAL] 2D-arrangement
Find Differences:
1. Construction of half-edges, univariate representation of the curves, instead of line swipe.
2. Find successor half-edge

>>> [subdivision:tool]
curves = extract_curves_from(map/image)
GUI:tab1 - dominant orientation detection
GUI:tab2 - lines/rays/segments
GUI:tab3 - circles/arcs
GUI: connected component detection -> find connectivity between segments/rays/arcs/...






[v] Pass Configuration as a dictionary {Multiprocessor, Endpoint_inclusion}




--------------------------------------------------------------------------------
Debugging:
----------
[v] graphs are stored in the decomposistion class, do we need to store them separately?
- [x] remove: self.MDG ( is included in self.decomposition )
  [v] remove: self.subGraphs ( are included in self.subDecomposition )



[later] fix the  - test_cases_key[13]
it happens to be genuinely a degenerate case, that is to say the peoblem 
happens when the angles of tangent to two tangent half-edge differ after
the 7th digit of the fractional part

[vx] the angle restriction in arc description

[vx] if a bunch of curves intersect in a way that there are more than one nodes, it will result in at least one half-edge. consequently the face identification will initiate, but if there is no face, it will become problematic.
TODO: Of each subgraph construct a graph, only of positive half edges, start face identification only if there is a close path in it.
What I Have Done: added the following to the find_successor_...
if len(candidateEdges) == 0: return allHalfEdgeIdx.index(twinIdx)
this unforunately will result in a having faces with null area
for instance, if a subgraph contains no cycle (i.e. a tree)

[v] decomposition.find_neighbours()
does not check the holes in both direction

[v] test_cases_key[23]
the superFace is not identified correctly!
hypothesis:
when a subdivision returns only two face, of which one is superFace
the sum of the internal angles of superFace is always bigger than 
the corresponding value of the inner face.

[v] test_cases_key[23]
what I define as the side of the half-edge, is actually the direction of
the half-edge with respect to the theta value of the underlying curve
Using it as side, casued a big trouble in finding the superFace,

in Half-edge class: change the attribute side to direction

for sd in mySubdivision.subDecompositions:
print '-----------'
print 'the face:', [ mySubdivision.MDG[s][e][k]['obj'].side
for (s,e,k) in sd.faces[0].halfEdges ]
print 'the superFace:', [ mySubdivision.MDG[s][e][k]['obj'].side
for (s,e,k) in sd.superFace.halfEdges ]

the face: ['positive', 'positive', 'negative', 'negative']
the superFace: ['positive', 'positive', 'negative', 'negative']
-----------
the face: ['positive', 'positive', 'negative', 'negative']
the superFace: ['positive', 'positive', 'negative', 'negative']
-----------
the face: ['positive', 'positive', 'positive', 'negative']
the superFace: ['positive', 'negative', 'negative', 'negative']

[v] fix the sorting bug
- test_cases_key[2]
  test_cases_key[9]

[v] fix the concentric circles problem:
- test_cases_key[7]
  the problem is that two holes are inside each other
  although it won't affect the point_in_face operation,
  it does mess up the visulization.
  it was due to the nested holes

[v] the wrong while loop condition - update subdivision over comments from Slawomir

[v] do we still have to start with all lines and then circles?
    or was it a constraint of the first approach?
    it's important to figure out as it would allow the update with any given
    function? otherwise with new lines everything should be computed from scratch!
    - the answer is it doesn't matter

[v] how to include those unbounded regions (oo, -oo) - should we?
a region unbounded to infinity is a degenerate case for pointInPolygone!
I think if i want to include the whole space, ie. faces unbounded to infinity
there should be infinity points oneach line, not only two and shared.
- the answer is treat all exterior region as one,
  so if a point is not in any face, it outside!
  
[v] move "face_2_path" to classmethod of subdivision

[v] decomposition problems:

[v]  problem #1
should I remove the infinity points?
NO! they are not the problem! because the values to identify the next half-edge
comes from the functions at the intersection points, has nothing to do with infinity
we actually need them, because such faces will be rejected based on that
actually yes! I had to remove 

[v] problem#2
is this related to the second key that I compute? (cross/project)
it seems there is something fishy there. check out the mode=1/2

[v] problem#3
the calculation of the first derivative on the line class is wrong,
not anymore, hopefuly

[v] remove infinity points/node
they overcomplicate things are not useful any way
and makes the update of the subdivision with new function way messy
remove and test with only lines.

[v] find_face_of_point doesn't work properly, why?
because we include the superFace, it returns the superFace in most cases
a crossing at an intersection of multiple half edges will be counted as multiple crossing!
I'm gonna most likely use the matplolib anyway

[v] how to detect the superFace? -step four
for now I decided to prevent its creation, instead of detection
but, half-edges connected to infinity nodes are not listed in "allhalfEdgeIdx"
therefore step one of decompose() practically does nothing!
consequently it is not possible to prevent the creation of superFace there.
it was due to isinstance(f, sym.Line) instead of (f.obj, sym.Line)
in the "find_functions_of_intersectionPoints()"
at this point there is no superFace

[v] Stand Alone Functions: handling the non-intersecting circles (in: decompose)
unbounded functions are connected to nodes in infinity,
so they will be handles otherwise.
Flag functions of bounded class (e.g. circles) that do not intersect
with any other functions. Those that are augmented with an arbitrary
point instead of an intersection. Through the decomposition process
skip those flagged.

[v] circle with r = 0 is problematic! look into that.

[v] be careful with r<0, sympy accepts it and things get messy!

[v] I think 2ndDer does not need change of direction, only 1stDer.
be careful, the angles between 1st and 2nd derivatives of inside half-edges
are pi/2, but the same angles for outside half-edges are -pi/2

[v] fix the __class__ problem in modilfied geometric instances
(inheritance -> aggregation)

[v] floatting point problem for the tangent intersection point, not rejected!
(in: merge_collocated_intersectionPoints)

[v] plotting: the problem of pathes-pathces
it seems to be related to the ordering of the starting and ending point of the arc

--------------------------------------------------------------------------------
Developement
------------

[ ] testing unit
- [v] create test cases for the improved cases, i.e. ray, segment, arc	
  [] use suite() so it runs all tests, even if one fails
  [] bring all "asserts" from subdivision.py to test.py

[v] merging and splitting faces locally.

[ ] should I have used  "geometric_graph"?


[ ] 'save_to_image(fileName)'
it should be fast, both for debugging sessions' sake and final application

[ ] multi-level of abstraction, wrt functions's priority
like the functions could be in 3 groups, [H]igh, [M]iddle, and [L]ow priority
and the subdivision would be with 3 levels of abstraction
subdivision.graphs['H'].mdg (based on functions from [H] priority list)
subdivision.graphs['M'].mdg (based on functions from [M] priority list)
subdivision.graphs['L'].mdg (based on functions from [L] priority list)
subdivision.graphs['A'].mdg (based on functions from [H,M,L] priority lists)

[ ] what does "http://toblerity.org/shapely/manual.html" do?

[ ] https://www.toptal.com/python/computational-geometry-in-python-from-theory-to-implementation

[v] decomposition.does_intersect() (for agents tracking)
just check their superface, to see whether they intersect or not!


[v] Subdivision.transform()
- [v] modify LineModified (to update: self.obj) --> RayModified and SegmentModified should inherit
  [v] CircleModified (to update: self.obj)
  [v] ArcModified (to update: self.obj / self.t1, self.t2)
  [v] update decomposition.faces (to update: self.path / self.holes[i].path)
  [v] update subdivision.curves --> should propagate to decomposition.curves
  [v] update subdivision.graph (nodes)  --> should propagate to decomposition.graph

[v] visualize the test cases without deploying subdivision

[v] arc
the challenge in handling the Arc is due to the fact that Arc is not
native to sympy. I have to use circle class for the internal object
in aggregation. for instance following methods of sympy would not work!
- sym.intersection( obj1, obj2 )
  sym.are_similar( obj1, obj2 )
  
  short term solution:
  - class LineModified(sym.Line):	  __class__ = sym.Line
    
    long-term solutions:
    - http://docs.sympy.org/latest/modules/geometry/entities.html#sympy.geometry.entity.GeometryEntity
      - http://docs.sympy.org/latest/modules/geometry/curves.html
	
    procedure to modify the code:
  1) add segment type to where ever I use "isinstance"
     - [] intersection_star (problem: sym.intersection)
       [] store_curves (problem: sym.are_similar) 
       - construct_nodes
	 step 2: reject an intersection if it is not a point
	 step 3: handling non-intersecting Curves
	 - construct_edges
	   step 3: half-edge construction
	 - edgeList_2_mplPath
	   fortunately path.arc() exists :) stupid sympy!

[v] add line segments and rays
procedure to modify the code:
add segment type to where ever I use "isinstance"
- construct_edges
  step 3: half-edge construction
- edgeList_2_mplPath
  fortunately path.arc() exists :) stupid sympy!


[v] Decomposition.find_neighbours(),
don't forget to include half-edges from holes.
for this actually we need to make sure there is no redundancy in holes!

[v] Subdivision.csvParser

[v] punch hole

[v] plot faces with patches

[v] remove infinity points

[v] detect the superFace of the subgraph:
find convex hull of all points in the subgraph - sym.convex_hull()
start from one of the points in the convex hull,
and follow the same procedure of face finding, with inverse sorting.      

[v] should I use mpl.path?
path.arc()                      # of a unit circle
path.circle()                   # 
path.intersects_bbox()          # 
path.intersects_path()          # 
path.contains_point()           # 
path.contains_path()            # 
path.contains_points()          # 

--------------------------------------------------------------------------------

clean-up, and speed-up
----------------------
[ ] clean up subdivision._decompose(self)

[ ] supposedly quick
Now that so many of the self.variables are removed from self., each method, such as construct_nodes() and construct_edges(), could be improved. they were horrible, very index dependant mostly because of these ipsCurveTVal, ipsCurveIdx, curveIpsTVal, curveIpsIdx. Generate nodes and edges right after computing their values, instead of storing in a list.


[ ] Do I actually need to update tvals of nodes and edges after transformation? I need them only for comparison with each other and they must be relatively correct in the original reference frame. However, if an operation comes after, eg face_merge, which needs new tvals, this could be problematic. Does face merge or any operation depends on that? Where do I actually need them?
- construct_edge()
  edgeList_to_mplPath() -> for face path construction including circles
  in plotting circles


[ ] shall I move those methods that are not useful after decomposition? No! just make them internal
- edgeList_to_mplPath
- store_curves
- find_successor_halfEdge
- decompose_graph
- construct_nodes
- construct_edges

- Slawomir: starting the name of a method with an underscore is a convention to mark them as internal.

http://stackoverflow.com/questions/551038/private-implementation-class-in-python
Use a single underscore prefix:
This is the official Python convention for 'internal' symbols;
"from module import * " does not import underscore-prefixed objects.

[ ] from sympy.geometry import Curve

[ ] half edge attributes:
- [v] sIdx, eIdx are redundant, they should be the same as selfIdx[0], selfIdx[1]
- [v] 1stDer, 2ndDer removed.
- [v] TVal  needed? - something is wrong in visualization! I don't know what!
  -> I don't have to use IPE for eTVal and sTVal, just need to fetch them from the node!
  
[ ] pycuda?

[ ] should I switch from sympy to CGAL?

[ ] intersection is the bottle-neck - how to improve that?

[ ] merge_collocated_intersectionPoints - too slow!

[ ] caching - store sorted outlets from each nodes

[ ] pointInPolygon - speed up

[ ] index-dependant implementation is a recepie for disaster
at least delet every temporarly varibale right after it's done its job





[v] remove self.MDG and self.subGraphs
[v] do I need the subdivision class? YES!

[v] get away from the spaghetti style - don't store self.variables
- [v] self.intersectionsFlat
  [v] self.intersections
  [v] self.ipsCurveTVal
  [v] self.ipsCurveIdx
  [v] self.curveIpsTVal
  [v] self.curveIpsIdx
  [v] self.edges
  [v] standAloneCurvesIdx

[v] do we need the Node class? [I actually had to bring the Node class back!]
isn't the dictionary of the MDG.node[0]
to write:
point = subdiv.MDG.node[nodeIdx]['point']
instead of:
point = subdiv.MDG.node[nodeIdx]['obj'].point

[v] nodes[idx] = (idx,nodeObject)
	I had this format of tuple, because it was required by the networkX
	TODO: 
	- [v] remove self.nodes[idx]
	  [v] subdiv.nodes[idx][1]['obj'] -> subdiv.MDG.node[idx]['obj']

[v] remove derivatives
since we use tangent and curvature for sorting, there is no longer a need for 
the derivatives. remove all related values from subdivision class

[v] inheritance VS. aggregation - Modified Geometry Instances

[v] inheritance VS. aggregation - Subdivision

[v] instantiation
examine all the internal variables of the subdivision class,
and see whether if they should belong to a class of their own.
- [v] nodes
  [v] edges
  [v] faces

[v] multi-processing


--------------------------------------------------------------------------------
Documentation
-------------
      [ ] add parser's manual to the readMe file

      [ ] API documentation
	$ cd Dropbox/myGits/dev/subdivision/
	$ pyreverse -o svg -p subdivision src/*.py

	-a N, -A    depth of research for ancestors
	-s N, -S    depth of research for associated classes
	-A, -S      all ancestors, resp. all associated
	-m[yn]      add or remove the module name
	-f MOD      filter the attributes : PUB_ONLY/SPECIAL/OTHER/ALL
	-k          show only the classes (no attributes and methods)
	-b          show 'builtin' objects

	- what are the date structures?
	  - a tree of data structures; e.g
	    subdiv: (MDG, decomposition, [curves], ... )
	    decomposition: (graph, [faces], ... )
	    MDG: (nodes, edges)
	  
	- how to index each data structure and access their object?
	  
	- what is the relation between indices of different lists,
	  e.g. nodes vs ips vs edges ...

      [ ] update report over
	- comments from Adam and Slawomir
	- sorting procedure 1st-2nd derivatives -> tangentAngle and curvature


https://readthedocs.org/

[ ] doc-tool:
- https://github.com/networkx/networkx/blob/master/networkx/classes/digraph.py
  https://docs.python.org/devguide/documenting.html
  http://docutils.sourceforge.net/
  http://docutils.sourceforge.net/rst.html
  http://www.sphinx-doc.org/en/stable/

[ ] examples

[ ] GUI?
- Load file / interactive drawings / manual entering
  Support animation

[ ] subdivision.io
https://upload.wikimedia.org/wikipedia/commons/8/88/Doubling_time_vs_half_life.svg

  
[v] animation
- https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
  https://jakevdp.github.io/blog/2013/05/28/a-simple-animation-the-magic-triangle/
  http://jakevdp.github.io/blog/2013/05/19/a-javascript-viewer-for-matplotlib-animations/
[gif] convert -delay 10 -loop 0 *.png animation.gif
[mov] ffmpeg -framerate 1/2 -i img%04d.png -c:v libx264 -r 30 out.mp4
  


