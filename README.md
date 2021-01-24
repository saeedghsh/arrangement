Arrangement
===========

A python package for 2D arrangement.
<p align="center">
	<img src="https://github.com/saeedghsh/arrangement/blob/master/docs/animation.gif" width="250">
</p>

This package has been developed to be employed as the underlying spatial representation for robot maps in the following publications:
- Saeed Gholami Shahbandi, Björn Åstrand and Roland Philippsen, *Sensor based adaptive metric-topological cell decomposition method for semantic annotation of structured environments*, ICARCV, Singapore, 2014, pp. 1771-1777. doi: 10.1109/ICARCV.2014.7064584 [URL](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7064584&isnumber=7064265).
- Saeed Gholami Shahbandi, Björn Åstrand and Roland Philippsen, *Semi-supervised semantic labeling of adaptive cell decomposition maps in well-structured environments*, ECMR, Lincoln, 2015, pp. 1-8. doi: 10.1109/ECMR.2015.7324207 [URL](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7324207&isnumber=7324045).
- Saeed Gholami Shahbandi, *Semantic Mapping in Warehouses*, Licentiate dissertation, Halmstad University, 2016. [URL](http://urn.kb.se/resolve?urn=urn:nbn:se:hh:diva-32170)
- Saeed Gholami Shahbandi, Martin Magnusson, *2D Map Alignment With Region Decomposition*, CoRR, abs/1709.00309, 2017. [URL](https://arxiv.org/abs/1709.00309)


IMPORTANT NOTE: This is an experimental implementation and it is HEAVILY UNDER MAINTENANCE.
Currently, only straight lines and circles are supported.
For a stable, fast and reliable implementation of the arrangement, I recommend the [CGAL](http://doc.cgal.org/latest/Arrangement_on_surface_2/index.html) library, although, CGAL is written in C++ and its [binding](https://github.com/CGAL/cgal-swig-bindings/wiki) does not include the arrangement package.

Dependencies and Download
-------------------------
- Download, installing dependencies, and install package
```shell
# Download
$ git clone https://github.com/saeedghsh/arrangement.git
$ cd arrangement

# Install dependencies
~$ pip install -r requirements.txt # python 2~
~$ pip3 install -r requirements.txt # python 3~
cat requirements.txt | xargs -n 1 pip install -Iv

# Install the package [optional]
python setup.py install # python 2
python3 setup.py install # python 3
```

- Demo
```shell
$ python demo.py --file_name 'tests/testCases/example_01.yaml' --multiprocessing 4
$ python3 demo.py --file_name 'tests/testCases/example_01.yaml' --multiprocessing 4
```

Basic Use and API
-----------------
- Basic Use
```python
>>> import sys
>>> sys.path.append('path_to_arrangement_repo') # if the package is not installed

# define curves:
>>> import arrangement.geometricTraits as trts
>>> traits = [trts.CircleModified( args=((i,i), 3) ) for i in range(4)]

# deploy arrangement
>>> import arrangement.arrangement as arr
>>> arrang = arr.Arrangement(traits, multiProcessing=4)

# visualize the result
>>> import arrangement.plotting as aplt
>>> aplt.animate_face_patches(arrang)
```

<p align="center">
	<img src="https://github.com/saeedghsh/arrangement/blob/master/docs/animation.gif">
</p>

- Arrangement class Hierarchy (the figure is created by [Pyreverse](https://www.logilab.org/blogentry/6883))
![classes_arrangement](https://github.com/saeedghsh/arrangement/blob/master/docs/classes_arrangement.png)

- Accessing nodes, edges and faces
```python
for nodeIdx in arrange.graph.nodes():
    print (nodeIdx, ': ', arrange.graph.node[nodeIdx]['obj'].attributes)

for halfEdgeIdx in arrange.graph.edges(keys=True):
    (s,e,k) = (startNodeIdx, endNodeIdx, path) = halfEdgeIdx
    print ( (s,e,k), ': ', arrange.graph[s][e][k]['obj'].attributes )

for fIdx,face in enumerate(arrange.decomposition.faces):
    print (fIdx, ': ', face.attributes)
```

- Visualization, plotting nad animating
```python
aplt.plot_decomposition(arrang,
                        interactive_onClick=False, interactive_onMove=False,
                        plotNodes=True, printNodeLabels=True,
                        plotEdges=True, printEdgeLabels=True)

aplt.animate_face_patches(arrang, timeInterval = .5*1000)
```

- Transformation example
```python
# arrange.transform_sequence('sequence', ( values, ), ( point, ) )
arrange.transform_sequence('T', ( (10,0), ), ( (0,0), ) )
arrange.transform_sequence('R', ( np.pi/2, ), ( (0,0), ) )
arrange.transform_sequence('S', ( (.2,.2), ), ( (0,0), ) )
arrange.transform_sequence('SRT', ((5,5), -np.pi/2, (-10,0), ),
                                 ((0,0), (0,0),    (0,0), ) )
```
<!-- ![translate](https://github.com/saeedghsh/arrangement/blob/master/docs/T.png) -->
<!-- <translate src="https://github.com/saeedghsh/arrangement/blob/master/docs/T.png" alt="none" width="50" height="50"> -->
<!-- ![rotate](https://github.com/saeedghsh/arrangement/blob/master/docs/R.png) -->
<!-- <rotate src="https://github.com/saeedghsh/arrangement/blob/master/docs/R.png" alt="none" width="50" height="50"> -->
<!-- ![scale](https://github.com/saeedghsh/arrangement/blob/master/docs/S.png) -->
<!-- <scale src="https://github.com/saeedghsh/arrangement/blob/master/docs/S.png" alt="none" width="50" height="50"> -->
<!-- ![SRT](https://github.com/saeedghsh/arrangement/blob/master/docs/SRT.png) -->
<!-- <SRT src="https://github.com/saeedghsh/arrangement/blob/master/docs/SRT.png" alt="none" width="50" height="50"> -->

- Storing curves in a yaml file.
A yaml file storing the curves should look like this:
```yaml
lines:
    - [x1,y1, x2,y2] or [x1,y1, slope]
segments:
    - [x1,y1, x2,y2]
rays:
    - [x1,y1, x2,y2]
circles:
    - [center_x, center_y, radius]
arcs:
    - [center_x, center_y, radius, interval_lower , interval_upper]
```
Note: ```arc``` is not tested and I suspect there are degenerate cases that are not handled properly.

<!-- boundary: -->
<!-- 	- [xMin, yMin, xMax, yMax] -->
<!-- ``` -->

See examples of yaml files in [testCases](https://github.com/saeedghsh/arrangement/tree/master/tests/testCases).
Use the script [utils.py](https://github.com/saeedghsh/arrangement/blob/master/arrangement/utils.py) to retrieve data from a yaml file as following:
```python
from arrangement.utils import load_data_from_yaml
data = load_data_from_yaml( address+fileName )
traits = data['traits']
```

<!-- - Checking sundivisions' intersection -->
<!-- ```python -->
<!-- import copy -->
<!-- arrang_copy = copy.copy(arrang) -->
<!-- arrang_copy.transform_sequence('R', ( np.pi/2, ), ( (0,0), ) ) -->
<!-- arrang_copy.transform_sequence('T', ( (-5,0), ), ( (0,0), ) ) -->

<!-- arrang_copy = copy.copy(arrang) -->
<!-- print arrange.decomposition.does_intersect(arrang_new.decomposition) -->
<!-- print arrange.decomposition.does_overlap(arrang_new.decomposition) -->
<!-- print arrange.decomposition.does_enclose(arrang_new.decomposition) -->
<!-- ``` -->


<!-- - Merging Faces -->
<!-- ```python -->
<!-- # arrange.merge_faces([face_indices,]) -->
<!-- arrange.merge_faces([2,3,4,5,6,7,8,9]) -->
<!-- aplt.animate_face_patches(arrang) -->
<!-- ``` -->
<!-- ![merge_faces](https://github.com/saeedghsh/arrangement/blob/master/docs/merge_faces.png) -->
<!-- <\!--- <merge_faces src="https://github.com/saeedghsh/arrangement/blob/master/docs/merge_faces.png" alt="none" width="50" height="50"> --\-> -->

<!-- Face merging is based on the indices of face and after merging faces, the decomposition changes and indices are altered. -->
<!-- Therefore, if it is desired multiple set of faces, identified by their indices, to be merged, it should be done as following: -->

<!-- ```python -->
<!-- # arrange.merge_faces([ [face_indices,], [face_indices,], ]) -->
<!-- arrange.merge_faces([[2,3,4,5,6,7],[8,9]]) -->
<!-- ``` -->

<!-- - Acessing half-edge of the outer boundary -->
<!-- ```python -->
<!-- outer_halfedge_idx = arrange.get_boundary_halfedges() -->
<!-- ``` -->


<!-- For more examples and details see the [demo.py](https://github.com/saeedghsh/arrangement/blob/master/demo.py). -->

Laundry List
------------
- [ ] trts.ArcModified (obj.center.x:sympy, obj.center.y:sympy, obj.radius:sympy, t1:numpy, t2:numpy)
  sympy objects have evalf() method that returns "normal" numbers, and numpy instances don't have it.
  this inconsistency might cause trouble later.
- [ ] documentation, and add more api examples.
- [ ] full test suite.
- [ ] fix known bugs.
- [ ] profile for speed-up.
- [ ] svg parsing is incomplete and disabled.
- [ ] clean up the ```plotting.py```
- [ ] storage to XML, compatible with [IEEE Standard for Robot Map Data Representation](http://ieeexplore.ieee.org/document/7300355/).
	- ```utilts.save_to_xml(file, myArrange)```
	- ```myArrange = utilts.load_from_xml(file)```
- [x] python3 compatible.
- [x] ```setup.py```


License
-------
Distributed with a GNU GENERAL PUBLIC LICENSE; see [LICENSE](https://github.com/saeedghsh/arrangement/blob/master/LICENSE).
```
Copyright (C) Saeed Gholami Shahbandi
```
