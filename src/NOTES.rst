TODO:
-----

      Debugging:
      ----

      [ ] fix the  - test_cases_key[13]
      it happens to be genuinely a degenerate case, that is to say the peoblem 
      happens when the angles of tangent to two tangent half-edge differ after
      the 7th digit of the fractional part


      [v] fix the sorting bug
      - test_cases_key[2]
	test_cases_key[9]

      [v] fix the concentric circles problem:
      - test_cases_key[7]
	the problem is that two holes are inside each other
	although it won't affect the point_in_face operation,
	it does mess up the visulization.
	it was due to the nested holes

      [v] Subdivision.csvParser
      [v] the degenerate case - update subdivision over comments from Slawomir
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
      [x] find_face_of_point doesn't work properly, why?
	  [ ] because we include the superFace, it returns the superFace in most cases
	  [ ] a crossing at an intersection of multiple half edges will be counted as multiple crossing!
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





      Developement
      ------------

      [ ] create test cases for the improved cases, i.e. ray, segment, arc

	add line segments and rays
	  the challenge is in handling the Arc which is not native to sympy
	  I have to use circle class for the internal object in aggregation
	  for instance following methods of sympy would not work!
	  - sym.intersection( obj1, obj2 )
	    sym.are_similar( obj1, obj2 )

	  class LineModified(sym.Line):
	  __class__ = sym.Line

	  look at these for potential (though long-term) solutions
	  - http://docs.sympy.org/latest/modules/geometry/entities.html#sympy.geometry.entity.GeometryEntity
	  - http://docs.sympy.org/latest/modules/geometry/curves.html

	  procedure to modify the code:
	  1) add segment type to where ever I use "isinstance"
	     - intersection_star (problem: sym.intersection)
	     - store_curves (problem: sym.are_similar) 
	     - construct_nodes
	       step 2: reject an intersection if it is not a point
	       step 3: handling non-intersecting Curves
	     - construct_edges
	       step 4: half-edge construction
	     - edgeList_2_mplPath
	       fortunately path.arc() exists :) stupid sympy!

	  2) when dealing with ray or segment (and later Arc), a given point might not be on the object (out of the interval). that's why I should always check if object.contians(point) this appears in IPE,DPE,... so, whenever using those, make sure to consider the cases where these methods might return False instead of expected type.

      [ ] testing unit
      use suite() so it runs all tests, even if one fails

      [ ] Multiple subdivision intersection (for agents tracking)

      [ ] Subdivision.transform(M(R,T,S))
	- Essential for the dynamic subdivision
	- Robot-centric subdivision [1]
	- extended perception field [2]
	  [1] Robot-centric subdivision (real time)
	  Subdivision of what ever the robot sense inside a circle with a radius of the sensor range.
	  [2] extended perception field via *local* communication between agents
	    - requires local map merging from different agents (subdivision matching)
	    - maybe using signal amplifiers to enable agents' communication

      [ ] Dynamic Subdivision - self.update_with_new_functions([newFunctions])

      [ ] also look into the "constructive geometry", merging and splitting faces locally.

      [ ] 'save_to_image(fileName)'
	it should be fast, both for debugging sessions' sake and final application

      [ ] check if the point is on any of the border functions
	how does path.contains_point(p) work? -> it checks contain, not enclose!
	includes the path itself, or not?

      [ ] multi-level of abstraction, wrt functions's priority
	like the functions could be in 3 groups, [H]igh, [M]iddle, and [L]ow priority
	and the subdivision would be with 3 levels of abstraction
	subdivision.graphs['H'].mdg (based on functions from [H] priority list)
	subdivision.graphs['M'].mdg (based on functions from [M] priority list)
	subdivision.graphs['L'].mdg (based on functions from [L] priority list)
	subdivision.graphs['A'].mdg (based on functions from [H,M,L] priority lists)

      [ ] what does "http://toblerity.org/shapely/manual.html" do?

      [ ] https://www.toptal.com/python/computational-geometry-in-python-from-theory-to-implementation

      [v] Decomposition.find_neighbours(),
	don't forget to include half-edges from holes.
	for this actually we need to make sure there is no redundancy in holes!

      [v] intersect subgraphs
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








      clean-up, and speed-up
      ----------------------
      [ ] half edge attributes:
	- [v] sIdx, eIdx are redundant, they should be the same as selfIdx[0], selfIdx[1]
	- [v] 1stDer, 2ndDer removed.
	- [] TVal  needed? - something is wrong in visualization! I don't know what!
	  -> I don't have to use IPE for eTVal and sTVal,
	  -> just need to fetch them from the node!
	  -> we have the node, and the current curve, easily find the corresponding tVal

      [ ] do we need the node class?
	isn't the dictionary of the MDG.node[0]
	to write:
	point = subdiv.MDG.node[nodeIdx]['point']
	instead of:
	point = subdiv.MDG.node[nodeIdx]['obj'].point

      [ ] remove self.intersectionsFlat (and other unused), after decomposition is done

      [ ] API documentation
	$ cd Dropbox/myGits/dev/subdivision/
	$ pyreverse -o svg -p subdivision src/*.py


	important note about nodes of networkx:
	MDG.nodes() is not neccessarily [0,1,2,...]
	it's important to remember that MDG.node is a dict, not a list
	MDG.node[idx] is not actually indexing the MDG.node, but fetching from a dict

	access nodes of networkX graph:
	for nodeIdx in subdiv.MDG.nodes():
	    print subdiv.MDG.node[nodeIdx]

	access "node instances" of nodes:
	for nodeIdx in subdiv.MDG.nodes():
	    print subdiv.MDG.node[nodeIdx]['obj']

	access edges of networkX graph:
	for halfEdgeIdx in subdiv.get_all_HalfEdge_indices():
	(s,e,k) = (startNodeIdx, endNodeIdx, path) = halfEdgeIdx
	    print subdiv.MDG[s][e][k]

	access "half-edges":
	for halfEdgeIdx in subdiv.get_all_HalfEdge_indices():
	    (s,e,k) = (startNodeIdx, endNodeIdx, path) = halfEdgeIdx
	    print subdiv.MDG[s][e][k]['obj']


	- what are the date structures?
	  - a tree of data structures; e.g
	    subdiv: (MDG, decomposition, [curves], ... )
	    decomposition: (graph, [faces], ... )
	    MDG: (nodes, edges)
	  
	- how to index each data structure and access their object?
	  - nodes: subdivision.nodes[idx], where else are they accessible?
	    edges: subdivision.edges[idx], where else are they accessible?
	    faces: subdivision.decomposition.faces[idx]
	    half-edges: subdivision.MDG[s][e][k]
	    ...
	  
	- what is the relation between indices of different lists,
	  e.g. nodes vs ips vs edges ...

      [ ] add parser's manual to the readMe file

      [ ] update report over
	- comments from Adam and Slawomir
	- sorting procedure 1st-2nd derivatives -> tangentAngle and curvature

      [ ] pycuda?

      [ ] should I switch from sympy to CGAL?

      [ ] intersection is the bottle-neck - how to improve that?
      [ ] merge_collocated_intersectionPoints - too slow!

      [ ] caching - store sorted outlets from each nodes

      [ ] pointInPolygon - speed up

      [ ] index-dependant implementation is a recepie for disaster
      at least delet every temporarly varibale right after it's done its job

      [v] nodes[idx] = (idx,nodeObject)
	obviously the idx in the tuple is redundant
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
	[v] nodes
	[v] edges
	[v] faces

      [v] multi-processing



Documentation
-------------
[ ] Parametric Equations

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
  
[v] animation
- https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
  https://jakevdp.github.io/blog/2013/05/28/a-simple-animation-the-magic-triangle/
  http://jakevdp.github.io/blog/2013/05/19/a-javascript-viewer-for-matplotlib-animations/
[gif] convert -delay 10 -loop 0 *.png animation.gif
[mov] ffmpeg -framerate 1/2 -i img%04d.png -c:v libx264 -r 30 out.mp4
  
[ ] subdivision.io

https://upload.wikimedia.org/wikipedia/commons/8/88/Doubling_time_vs_half_life.svg



Abstract and other notes
--------
What are the differences between this method and straight line subdivision?
First of all, both use DCEL data structure for the spatial representation of the subdivision
Therefore they share the same method for finding neighboring faces

The angle to the next vertex in the conventional method is
equal to the angle of the edge in between the two following vertices, also
equal to the gradient (tangent) of the line function in between.
However, in the presence of generic functions beyond straight lines,
the simple halfEdge's angle won't suffice finding closed loops.
This is due to the possibility of intersection points of tangent type.
One critical differences of the prosposed extension is to exploit
the value of the second derivative of the intersecting functions
at the tangent point, in order to ensure detection of simple closed loops (simple faces).
We will show that this approach is computationaly less expensive in comparison to
other alternatives, such as treating the subdivision as a graph and finding closed loops.

On the other, the presence of halfEdges which are not necessarily straight line segments,
invalidates the approach of detecting whether if a given point is inside a face based on
the cross-product of the inner-edges, and vectors from vertices to the given point.
In the extension, we adopt the pointInPolygon (PIP, aka "crossing number algorithm" or "even-odd rule") approach to answer the raised question.

The only restriction of the proposed method is that every class of functions to be included
must be expressable in terms of a single variable (namely \theta), and to be diffrensiable wrt to this variable (\theta) at any given point on the function's level curve.


NOTE: there is no exact representation of the circle using Bezier curves.

NOTE: the hole problem (a non-intersecting circle located inside anthor face)
If I had implemented the the "point In Polygone" as a classmethod, it could be sufficient to 
just add an extra edge (the whole circle) to the face.
But using mpl.path, such approach wouldn't work.
Therefore I add a new slot to the face class, that is a list of holes in each face. Each hole in the face.

NOTE:
I use mpl.path to represent faces
-> circles are approximated -> consequently find_face_point is also not accurate
Nevertheless, I do store edges in the face along with the path repesentation.
So anytime I manage to develope that stupid "point in polygone" with a decent speed,
this approximation won't be much of a problem.
