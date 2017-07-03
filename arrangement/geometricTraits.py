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

import numpy as np
import sympy as sym

################################################################################
################################################# classes aggregation from sympy
################################################# Modified Line
################################################# Modified Circle
################################################################################

note_on_DPE='''
It doesn't work for segment, and I guess Ray too.
I think It's because of the "contain" test, but not sure/.
''' 

note_on_isinstance = '''
Important note:
---------------
isinstance(circle, ArcModified) - > False
isinstance(circle, CircleModified) - > True

isinstance(arc, ArcModified) - > True
isinstance(arc, CircleModified) - > True

This is also true for the cases with arc, segment and line
isinstance(segment, SegmentModified) - > True
isinstance(segment, RayModified) - > False
isinstance(segment, LineModified) - > True

isinstance(ray, SegmentModified) - > False
isinstance(ray, RayModified) - > True
isinstance(ray, LineModified) - > True

isinstance(line, SegmentModified) - > False
isinstance(line, RayModified) - > False
isinstance(line, LineModified) - > True
'''

################################################################################
class LineModified:

    __doc__ = '''
    This class is to represent infinit line objects
    It is aggregated from sympy.Line class, hence the name.
    The sympy.Line is stored in self.obj
    It doesn't have any aditional parameter, but extra methods.

    usage:
    lm = LineModified( args= (X) )
    X is anything that could be passed to sympy.Line:
    X = ((x1,y1),(x2,y2))
    X = (sym.Point(x1,y1),sym.Point(x2,y2))
    X = ((x1,y1),slope)
    X = (sym.point(x1,y1),slope)

    {:s}

    {:s}
    '''.format(note_on_isinstance, note_on_DPE)

    ####################################
    def __init__ (self, args):
        '''
        LineModified class
        '''
        if isinstance(args[1],sym.Point):
            self.obj = sym.Line( *args )
        else:
            self.obj = sym.Line( args[0], slope=args[1] )

    ####################################
    def transform_sequence(self, operTypes, operVals, operRefs):
        '''
        LineModified class

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
        
        for opIdx, opType in enumerate(operTypes):
            
            if opType == 'T':# and all(operVals[opIdx]!=(0,0)):
                tx,ty = operVals[opIdx]
                self.obj = self.obj.translate(tx,ty)
                
            elif opType == 'R':# and operVals[opIdx]!=0:
                theta = operVals[opIdx]
                ref = operRefs[opIdx]
                self.obj = self.obj.rotate(theta,ref)
                
            elif opType == 'S':# and all(operVals[opIdx]!=(1,1)):
                sx,sy = operVals[opIdx]
                ref = operRefs[opIdx]
                self.obj = self.obj.scale(sx,sy,ref)

    ####################################
    def DPE(self, t):
        '''
        LineModified class

        TD: not sure when and how I did this, double-check

        Direct Parametric Equation
        (x,y) = df(t) = ( xl + a*t , yl + b*t )
        
        a**2 + b**2 = 1 --> b = sqrt(1-a**2)
        sl = b/a --> b = a *sl
        '''
        if self.obj.slope == 0:
            return sym.Point(self.obj.p1.x + t, self.obj.p1.y)
        elif self.obj.slope == sym.oo or self.obj.slope == -sym.oo:
            return sym.Point(self.obj.p1.x, self.obj.p1.y + t)
        else:
            a = sym.sqrt(self.obj.slope**2 + 1) / (self.obj.slope**2 + 1)
            b =  a * self.obj.slope
            return sym.Point(self.obj.p1.x + a.evalf()*t,
                             self.obj.p1.y + b.evalf()*t)

    ####################################
    def IPE(self, point):
        '''
        LineModified class

        TD: not sure when and how I did this, double-check

        Inverse Parametric Equation
        t = df(x,y) = (x-xl)/a if a!=0 else (y-yl)/b
        
        a**2 + b**2 = 1 --> b = sqrt(1-a**2)
        sl = b/a --> b = a *sl
        '''
        if self.obj.slope == 0:
            return point.x
        elif self.obj.slope == sym.oo or self.obj.slope == -sym.oo:
            return point.y
        else:
            a = sym.sqrt(self.obj.slope**2 + 1) / (self.obj.slope**2 + 1)
            b = sym.sqrt(1 - a**2)
            return (point.x-self.obj.p1.x)/a.evalf() if a != 0 else (point.y-self.obj.p1.y)/b.evalf()

    ####################################
    def firstDerivative (self, point=None, direction='positive'):
        '''
        LineModified class

        generally:
        dy/dx = lim_{ Delta_x -> 0 } Delta_y / Delta_x
        
        for a straight line:
        dy/dx = Delta_y / Delta_x
        
        1stDer = [dx,dy],
        slope = tan(theta) -> theta = arctan(slope)
        ( note that theta \in [-pi/2, pi/2] )
        [dx, dy] = [cos(theta) , sin(theta)]
        ( note that always dx>=0, because theta \in [-pi/2, pi/2])

        note that slopeAngle is constant for the line
        and that's why the second derivative is a null vector
        '''
        slopeAngle = np.arctan(np.float(self.obj.slope.evalf()))
        res = np.array([np.cos(slopeAngle) , np.sin(slopeAngle)], float)
        return res if direction == 'positive' else -1*res


    ####################################
    def secondDerivative(self, point=None, direction='positive'):
        '''
        LineModified class

        although it is independant of the location of the point
        we include for consistency with a general form of traits
        note: direction does not affect the 2nd derivative of a line
        '''
        return np.array([0. ,0.], float)

    ####################################
    def tangentAngle(self, point=None, direction='positive'):
        '''
        LineModified class

        although it is independant of the location of the point
        we include for consistency with a general form of traits

        theta \in [0,2pi]
        '''
        (dx,dy) = self.firstDerivative(point, direction)
        alpha = np.arctan2(dy,dx)
        return np.mod(alpha + 2*np.pi , 2*np.pi)

    ####################################
    def curvature(self, point=None, direction='positive'):
        '''
        LineModified class

        although it is independant of the location of the point
        we include for consistency with a general form of traits
        note: direction does not affect the curvature of a line
        '''
        return 0.



################################################################################
#TODO:saesha
class RayModified(LineModified):

    __doc__ = '''
    This class is to represent a Ray objects (half-line)
    It is aggregated from sympy.Ray class, hence the name.
    It doesn't have any aditional parameter, but extra methods.

    note:
    -----
    that "RayModified" is a subClass of "LineModified"
    but the self.obj is aggregated from sym.Ray

    usage:
    ------
    rm = RayModified( args= (X) )
    X is anything that could be passed to sympy.Ray:
    X = ((x1,y1),(x2,y2))
    X = (sym.Point(x1,y1),sym.Point(x2,y2))
    X = ((x1,y1),slope)
    X = (sym.point(x1,y1),slope)
    Note that for sym.Ray, (x1,y1) is always the starting point of the ray

    {:s}
    '''.format(note_on_isinstance)

    ####################################
    def __init__ (self, args):
        '''
        RayModified class 

        note that "RayModified" is a subClass of "LineModified"
        but the self.obj is aggregated from sym.Ray
        
        '''
        self.obj = sym.Ray( *args )


    ####################################
    def DPE(self, t):
        '''
        RayModified class 

        TD: not sure when and how I did this, double-check

        Direct Parametric Equation
        (x,y) = df(t) = ( xl + a*t , yl + b*t )
        
        a**2 + b**2 = 1 --> b = sqrt(1-a**2)
        sl = b/a --> b = a *sl
        '''
        if self.obj.slope == 0:
            point = sym.Point(self.obj.p1.x + t, self.obj.p1.y)
            
        elif self.obj.slope == sym.oo or self.obj.slope == -sym.oo:
            point =  sym.Point(self.obj.p1.x, self.obj.p1.y + t)
        else:
            a = sym.sqrt(self.obj.slope**2 + 1) / (self.obj.slope**2 + 1)
            b =  a * self.slope #TODO:saesha, shouldn't this be a * self.obj.slope
            point =  sym.Point(self.obj.p1.x + a.evalf()*t,
                               self.obj.p1.y + b.evalf()*t)
        
        return point if self.obj.contains(point) else False
        
    ####################################
    def IPE(self, point):
        '''
        RayModified class 

        TD: not sure when and how I did this, double-check

        Inverse Parametric Equation
        t = df(x,y) = (x-xl)/a if a!=0 else (y-yl)/b
        
        a**2 + b**2 = 1 --> b = sqrt(1-a**2)
        sl = b/a --> b = a *sl
        '''

        # TODO: 
        # this condition checking is a problem when the trait and a node on it are rotated,
        # but after rotation, the node is slightly off the trait and this function would return False!
        # temporarily I disable this condition checking until later

        if True: #self.obj.contains(point):
            if self.obj.slope == 0:
                return point.x
            elif self.obj.slope == sym.oo or self.obj.slope == -sym.oo:
                return point.y
            else:
                a = sym.sqrt(self.obj.slope**2 + 1) / (self.obj.slope**2 + 1)
                b = sym.sqrt(1 - a**2)
                return (point.x-self.obj.p1.x)/a.evalf() if a != 0 else (point.y-self.obj.p1.y)/b.evalf()
        else:
            return False



################################################################################
#TODO:saesha
class SegmentModified(LineModified):

    __doc__ = '''
    This class is to represent a Segment objects (line-segment)
    It is aggregated from sympy.Segment class, hence the name.
    It doesn't have any aditional parameter, but extra methods.

    note:
    -----
    that "SegmentModified" is a subClass of "LineModified"
    but the self.obj is aggregated from sym.Segment

    usage:
    ------
    sm = SegmentModified( args= (X) )
    X is anything that could be passed to sympy.Segment:
    X = ((x1,y1),(x2,y2))
    X = (sym.Point(x1,y1),sym.Point(x2,y2))
    X = ((x1,y1),slope)
    X = (sym.point(x1,y1),slope)

    {:s}
    '''.format(note_on_isinstance)

    ####################################
    def __init__ (self, args):
        '''
        SegmentModified class 



        '''
        self.obj = sym.Segment( *args )


    ####################################
    def DPE(self, t):
        '''
        SegmentModified class 

        TD: not sure when and how I did this, double-check

        Direct Parametric Equation
        (x,y) = df(t) = ( xl + a*t , yl + b*t )
        
        a**2 + b**2 = 1 --> b = sqrt(1-a**2)
        sl = b/a --> b = a *sl
        '''
        if self.obj.slope == 0:
            point = sym.Point(self.obj.p1.x + t, self.obj.p1.y)
        elif self.obj.slope == sym.oo or self.obj.slope == -sym.oo:
            point =  sym.Point(self.obj.p1.x, self.obj.p1.y + t)
        else:
            a = sym.sqrt(self.obj.slope**2 + 1) / (self.obj.slope**2 + 1)
            b =  a * self.obj.slope #TODO:saesha, shouldn't this be a * self.obj.slope
            point =  sym.Point(self.obj.p1.x + a.evalf()*t,
                               self.obj.p1.y + b.evalf()*t)
        
        return point if self.obj.contains(point) else False
        
    ####################################
    def IPE(self, point):
        '''
        SegmentModified class

        TD: not sure when and how I did this, double-check

        Inverse Parametric Equation
        t = df(x,y) = (x-xl)/a if a!=0 else (y-yl)/b
        
        a**2 + b**2 = 1 --> b = sqrt(1-a**2)
        sl = b/a --> b = a *sl
        '''

        # TODO: 
        # this condition checking is a problem when the trait and a node on it are rotated,
        # but after rotation, the node is slightly off the trait and this function would return False!
        # temporarily I disable this condition checking until later

        if True: # self.obj.contains(point):
            if self.obj.slope == 0:
                return point.x
            elif self.obj.slope == sym.oo or self.obj.slope == -sym.oo:
                return point.y
            else:
                a = sym.sqrt(self.obj.slope**2 + 1) / (self.obj.slope**2 + 1)
                b = sym.sqrt(1 - a**2)
                return (point.x-self.obj.p1.x)/a.evalf() if a != 0 else (point.y-self.obj.p1.y)/b.evalf()
        else:
            return False
    

################################################################################
class CircleModified:

    __doc__ = '''
    This class is to represent circle objects
    It is aggregated from sympy.Circle class, hence the name.
    The sympy.Circle is stored in self.obj
    It doesn't have any aditional parameter, but extra methods.

    usage:
    lm = CircleModified( args= (X) )
    X is anything that could be passed to sympy.Circle:
    X = ((xc,yc),r)
    
    {:s}
    '''.format(note_on_isinstance)


    ####################################
    def __init__ (self, args):
        '''
        CircleModified class
        '''
        self.obj = sym.Circle( *args )

    ####################################
    def transform_sequence(self, operTypes, operVals, operRefs ):
        '''
        CircleModified class

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
        
        for opIdx, opType in enumerate(operTypes):
            
            if opType == 'T':# and all(operVals[opIdx]!=(0,0)):
                tx,ty = operVals[opIdx]
                self.obj = self.obj.translate(tx,ty)
                
            elif opType == 'R':# and operVals[opIdx]!=0:
                theta = operVals[opIdx]
                ref = operRefs[opIdx]
                
                # Important note
                # ideally I would like to do this:
                # self.obj = self.obj.rotate(theta,ref)
                # but as rotate is not effective for circles (don't know why!)
                # and since I can not set the attributes of the circel as:
                # self.obj.center = self.obj.center.rotate(theta,ref)
                # I am left with no choice but to define a new circle and assign it to the obj
                # fortunately this won't be a problem, as this obj instance is an attribute
                # to the modifiedCircle instance, and that's what I want to maintain a reference to
                c = self.obj.center.rotate(theta,ref)
                r = self.obj.radius
                self.obj = sym.Circle(c,r)
                
            elif opType == 'S':# and all(operVals[opIdx]!=(1,1)):
                sx,sy = operVals[opIdx]
                ref = operRefs[opIdx]
                self.obj = self.obj.scale(sx,sy,ref)



    ####################################
    def DPE(self, t):
        '''
        CircleModified class

        Direct Parametric Equation
        (x,y) = fd(t)|(xc,yc,rc)
        '''
        fx = self.obj.center.x + self.obj.radius * np.cos(t) # sym.cos(t)
        fy = self.obj.center.y + self.obj.radius * np.sin(t) # sym.sin(t)
        return sym.Point( fx.evalf(), fy.evalf() )

    ####################################
    def IPE(self, point):
        '''
        CircleModified class

        Inverse Parametric Equation
        t = fi(x,y)|(xc,yc,rc)
        '''
        # if (point.y-self.center.y)**2 + (point.x-self.center.x)**2 == self.radius**2:
        dy = point.y - self.obj.center.y
        dx = point.x - self.obj.center.x
        return sym.atan2(dy, dx).evalf()
        
    ####################################
    def firstDerivative (self, point, direction='positive'):
        '''
        CircleModified class

        A circle's first derivative wrt \theta
        x = xc + radius*cos(theta)
        y = yc + radius*sin(theta)

        x' = -radius*sin(theta)
        y' = +radius*cos(theta)
        '''
        theta = np.float(self.IPE(point))
        x_ = -self.obj.radius * np.sin(theta) # sym.sin(theta)
        y_ = +self.obj.radius * np.cos(theta) # sym.cos(theta)
        res = np.array([x_.evalf(), y_.evalf()], float)
        return res if direction == 'positive' else -1*res            

    ####################################
    def secondDerivative(self, point, direction='positive'):
        '''
        CircleModified class

        A circle's second derivative wrt \theta
        x = xc + radius*cos(theta)
        y = yc + radius*sin(theta)

        x" = -radius*cos(theta)
        y" = -radius*sin(theta)

        note: direction does not affect the 2nd derivative of a circle

        '''
        theta = np.float(self.IPE(point))
        x_ = -self.obj.radius * np.cos(theta) # sym.cos(theta).evalf()
        y_ = -self.obj.radius * np.sin(theta) # sym.sin(theta).evalf()
        return np.array([x_.evalf(), y_.evalf()], float)

    ####################################
    def tangentAngle(self, point, direction='positive'):
        '''
        CircleModified class

        although it is independant of the location of the point
        we include for consistency with a general form of traits

        theta \in [0,2pi]
        '''
        (dx,dy) = self.firstDerivative(point, direction)
        alpha = np.arctan2(dy,dx)
        return np.mod(alpha + 2*np.pi , 2*np.pi)


    ####################################
    def curvature(self, point=None, direction='positive'):
        '''
        CircleModified class

        although it is independant of the location of the point
        we include for consistency with a general form of traits
        '''
        k = 1./self.obj.radius
        return k if direction == 'positive' else -1*k




################################################################################
class ArcModified(CircleModified):


    __doc__ = '''
    This class is to represent circle objects
    It is aggregated from sympy.Circle class, hence the name.
    The sympy.Circle is stored in self.obj
    It has two aditional parameters, and extra methods.
    additional parameters are:
    t1: the stating angle of the arc
    t2; the ending angle of the arc

    potential screw_up
    theta \in [-pi, pi])
    so how to define an arc where (t1,t2)=(pi/2, 3pi/2)
    well, best is to allow any interval ([-pi, pi] or [0, 2pi])
    and always check the tvalues +2pi and -2pi to see if whether they are
    in the range
    
    note:
    -----
    note that "ArcModified" is a subClass of "CircleModified"
    but the self.obj is aggregated from sym.Circle

    usage:
    ------
    pc, rc, interval =  (0,0), 1, (0,numpy.pi)
    am =  mSym.ArcModified( args=( pc, rc, interval )  )
    
    {:s}
    '''.format(note_on_isinstance)

    ####################################
    def __init__ (self, args):
        '''
        ArcModified class




        '''
        
        self.obj = sym.Circle( *args[:2] )

        # the radial distance of t1 to t2 can not be more than 2pi
        assert max(args[2])-min(args[2]) < 2*np.pi

        # make sure t1 and t2 are sorted CCW (t1<t2)
        self.t1 = min(args[2])
        self.t2 = max(args[2])

    ####################################
    def transform_sequence(self, operTypes, operVals, operRefs ):
        '''
        ArcModified class

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
        
        for opIdx, opType in enumerate(operTypes):
            
            if opType == 'T':# and all(operVals[opIdx]!=(0,0)):
                tx,ty = operVals[opIdx]
                self.obj = self.obj.translate(tx,ty)
                
            elif opType == 'R':# and operVals[opIdx]!=0:
                theta = operVals[opIdx]
                ref = operRefs[opIdx]

                # Important note
                # ideally I would like to do this:
                # self.obj = self.obj.rotate(theta,ref)
                # but as rotate is not effective for circles (don't know why!)
                # and since I can not set the attributes of the circel as:
                # self.obj.center = self.obj.center.rotate(theta,ref)
                # I am left with no choice but to define a new circle and assign it to the obj
                # fortunately this won't be a problem, as this obj instance is an attribute
                # to the modifiedCircle instance, and that's what I want to maintain a reference to
                c = self.obj.center.rotate(theta,ref)
                r = self.obj.radius
                self.obj = sym.Circle(c,r)

                # TODO: too many consequtive rotation might carry t1 and t2 out of the predefined interval
                # correct them if there is going to be any valid interval for t1 and t2
                self.t1 += theta
                self.t2 += theta

                
            elif opType == 'S':# and all(operVals[opIdx]!=(1,1)):
                sx,sy = operVals[opIdx]
                ref = operRefs[opIdx]
                self.obj = self.obj.scale(sx,sy,ref)
