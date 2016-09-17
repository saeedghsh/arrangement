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

################################################################################
################################################# classes aggregation from sympy
################################################# Modified Line
################################################# Modified Circle
################################################################################

################################################################################
#TODO:saesha
class RayModified:
    ####################################
    def __init__ (self, args):
        self.obj = sym.Ray( *args )

################################################################################
#TODO:saesha
class SegmentModified:
    ####################################
    def __init__ (self, args):
        self.obj = sym.Segment( *args )

################################################################################
#TODO:saesha
class ArcModified:
    ####################################
    def __init__ (self, args):
        self.obj = sym.Circle( *args )

        # make sure t1 and t2 are sorted CCW (t1<t2)
        self.t1 = None
        self.t2 = None






################################################################################
class LineModified:

    ####################################
    def __init__ (self, args):
        if isinstance(args[1],sym.Point):
            self.obj = sym.Line( *args )
        else:
            self.obj = sym.Line( args[0], slope=args[1] )

    ####################################
    def DPE(self, t):
        '''
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
            b =  a * self.slope #TODO:saesha, shouldn't this be a * self.obj.slope
            return sym.Point(self.obj.p1.x + a.evalf()*t,
                             self.obj.p1.y + b.evalf()*t)

    ####################################
    def IPE(self, point):
        '''
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
        although it is independant of the location of the point
        we include for consistency with a general form of curves
        note: direction does not affect the 2nd derivative of a line
        '''
        return np.array([0. ,0.], float)

    ####################################
    def tangentAngle(self, point=None, direction='positive'):
        '''
        although it is independant of the location of the point
        we include for consistency with a general form of curves
        '''
        (dx,dy) = self.firstDerivative(point, direction)
        alpha = np.arctan2(dy,dx)
        return np.mod(alpha + 2*np.pi , 2*np.pi)

    ####################################
    def curvature(self, point=None, direction='positive'):
        '''
        although it is independant of the location of the point
        we include for consistency with a general form of curves
        note: direction does not affect the curvature of a line
        '''
        return 0.
    

################################################################################
class CircleModified:

    ####################################
    def __init__ (self, args):
        self.obj = sym.Circle( *args )

    ####################################
    def DPE(self, t):
        '''
        Direct Parametric Equation
        (x,y) = fd(t)|(xc,yc,rc)
        '''
        fx = self.obj.center.x + self.obj.radius * np.cos(t) # sym.cos(t)
        fy = self.obj.center.y + self.obj.radius * np.sin(t) # sym.sin(t)
        return sym.Point( fx.evalf(), fy.evalf() )

    ####################################
    def IPE(self, point):
        '''
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
        # print self, theta, x_, y_
        return np.array([x_.evalf(), y_.evalf()], float)

    ####################################
    def tangentAngle(self, point, direction='positive'):
        '''
        although it is independant of the location of the point
        we include for consistency with a general form of curves
        '''
        (dx,dy) = self.firstDerivative(point, direction)
        alpha = np.arctan2(dy,dx)
        return np.mod(alpha + 2*np.pi , 2*np.pi)


    ####################################
    def curvature(self, point=None, direction='positive'):
        '''
        although it is independant of the location of the point
        we include for consistency with a general form of curves
        '''
        k = 1./self.obj.radius
        return k if direction == 'positive' else -1*k
