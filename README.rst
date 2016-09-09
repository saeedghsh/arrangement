Subdivision
===========
This repository provides a python package for decomposition of a 2D plane over a set of curves.
Currently straight lines and circles are the supported classes of curves.
please note that effiecency and optimallity have not been the objective of the implementation.
As a consequence, the code at your disposal is not in its best shape.
The objective of this implementation has been to show the concept and prototyping it.

.. image:: animation.gif


Download
--------

Dependencies:
* Python >=2.6
* numpy >= 1.10.2
* sympy >= 1.0
* networkx >= 1.10
* matplotlib >= 1.4.3

::
   git clone https://github.com/saeedghsh/subdivision/

Basic Use
---------
::
   import subdivision as sdv
   import sympy as sym
   import modifiedSympy as mSym
   import plotting as myplt
   
   # define curves:
   curves = []
   curves += [ mSym.LineModified( args=(sym.Point(0,0), sym.Point(1,1)) ) ]
   curves += [ mSym.CircleModified( args=(sym.Point(0,0), 1) ) ]
   
   # deploy subdivision
   mySubdivision = sdv.Subdivision(curves)

   # visualize the result
   myplt.animate_face_patches(mySubdivision)

for details see the `demo <https://github.com/saeedghsh/subdivision/src/demo.py>`   


Limitations and Bugs
--------------------
TODO


License
-------
Distributed with a BSD license; see LICENSE
::
   Copyright (C) Saeed Gholami Shahbandi <saeed.gh.sh@gmail.com>

