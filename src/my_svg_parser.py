import numpy as np
import xml.etree.ElementTree as ET
import yaml

################################################################################
def xml_tree_parser_to_svg_elements(tree):
    '''
    this function takes the xml tree of an SVG file
    parses, detects and returns elements of interest

    The element types that can cause graphics to be drawn onto the target canvas.
    ['path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'text', 'image', 'use']

    The <polyline> and <polygon> objects don't add anything that the more flexible path cannot do
    The <path> can be used to replace <rect>, <ellipse>, and <circle>

    <image xlink:href="p72.jpg" height="200" width="100" x="100" y="100"/>
    <text x="0" y="100" font-size="80" fill="red" > text_string < /text>
    '''
    res = {'path' : [],
           'circle' : [],
           'ellipse' : [],
           'rect' : [],
           'polyline' : [],
           'polygon' : [],
           'line' : [] }

    for element in tree.iter():

        tag = element.tag.split('}')[-1]
        if tag in res.keys():
            res[tag].append(element)
        
        # if 'path' in element.tag:
        #     res['path'].append(element)
        # elif 'circle' in element.tag:
        #     res['circle'].append(element)
        # elif 'ellipse' in element.tag:
        #     res['ellipse'].append(element)
        # elif 'rect' in element.tag:
        #     res['rect'].append(element)
        # elif 'polyline' in element.tag:
        #     res['polyline'].append(element)
        # elif 'line' in element.tag:
        #     res['line'].append(element)
        # elif 'polygon' in element.tag:
        #     res['polygon'].append(element)
        # else:
        #     pass # print element.tag

    return res

############################################################### parsing elements
def svg_parser_polyline_element(polyline_element):
    '''
    https://www.w3.org/TR/SVG/shapes.html#PolylineElement    
    '''
    # replacing all ',' with ' ' and splitting
    pts_str = polyline_element.attrib['points'].replace(',',' ').split(' ')

    # pairing points in the list and constructing the list on coordinates
    pts = [ [ float(pts_str[idx]), float(pts_str[idx+1]) ]
            for idx in range(0, len(pts_str), 2) ]

    return np.array( pts )

########################################
def svg_parser_polygon_element(polygon_element):
    '''
    https://www.w3.org/TR/SVG/shapes.html#PolygonElement
    '''
    # replacing all ',' with ' ' and splitting
    pts_str = polygon_element.attrib['points'].replace(',',' ').split(' ')

    # pairing points in the list and constructing the list on coordinates    
    pts = [ [ float(pts_str[idx]), float(pts_str[idx+1]) ]
            for idx in range(0, len(pts_str), 2) ]
    return np.array( pts )

########################################
def svg_parser_circle_element(circle_element):
    '''
    https://www.w3.org/TR/SVG/shapes.html#CircleElement    
    '''
    cx = float(circle_element.attrib['cx'])
    cy = float(circle_element.attrib['cy'])
    r = float(circle_element.attrib['r'])
    return np.array([cx, cy, r ])

########################################
def svg_parser_ellipse_element(ellipse_element):
    '''
    https://www.w3.org/TR/SVG/shapes.html#EllipseElement    
    '''
    cx = float(ellipse_element.attrib['cx'])
    cy = float(ellipse_element.attrib['cy'])
    rx = float(ellipse_element.attrib['rx'])
    ry = float(ellipse_element.attrib['ry'])
    return np.array([cx, cy, rx, ry])

########################################
def svg_parser_rect_element(rect_element):
    '''
    https://www.w3.org/TR/SVG/shapes.html#RectElement    
    '''
    x = float(rect_element.attrib['x'])
    y = float(rect_element.attrib['y'])
    w = float(rect_element.attrib['width'])
    h = float(rect_element.attrib['height'])
    return np.array([x, y, w, h])

########################################
def svg_parser_line_element(line_element):
    '''
    https://www.w3.org/TR/SVG/shapes.html#LineElement    
    '''
    x1 = float(line_element.attrib['x1'])
    y1 = float(line_element.attrib['y1'])
    x2 = float(line_element.attrib['x2'])
    y2 = float(line_element.attrib['y2'])
    return np.array([x1, y1, x2, y2])

########################################
def svg_parser_path_element(path_element):
    '''
    https://www.w3.org/TR/SVG/paths.html#PathElement

    Upper case -> absolute
    Lower case -> relative

    TODO:
    this method will, in the current form, ignore all curves, arcs, and their parameters.
    It just returns a numpy array, containing the points on the path
    [parameter-points of arcs and curves not included]

    Note:
    I have tried the implementation with indexing instead of (poping-deleting) the path_data,
    time improvement was from 2.95 to 2.65 on the scale of 100000 iterations,
    the time improvment this not worth sacrificing readability
    (though not much different in that perspective either)
    '''

    # replacing all ',' with ' ' and splitting
    path_data = path_element.attrib['d'].replace(',',' ').split(' ')

    pts = [  ]
    while len(path_data) > 0:

        # if the new item in the path_data is a character, it is stored in cmd
        # otherwise the command is what is already stored in cmd
        # i.e. if command is not explicitly stated, it is the same as the last.
        if not (path_data[0].replace('.','').replace('-','').isdigit()):
            cmd = path_data.pop(0)

        
        if cmd in ['A', 'a'] : #################################################
            [rx, ry, x_axs_rot, lrg_arc_flg, swp_flg, x, y] = [ float (c) for c in path_data[:7] ]
            del path_data[:7]

        elif cmd in ['C', 'c'] : ###############################################
            [x1, y1, x2, y2, x, y] = [ float (c) for c in path_data[:6] ]
            del path_data[:6]

        elif cmd in ['H', 'h'] : ###############################################
            x = float( path_data[:1] )
            del path_data[:1]
            y = pts[-1][1]

        elif cmd in ['L', 'l'] : ###############################################
            [x, y] = [ float (c) for c in path_data[:2] ]
            del path_data[:2]

        elif cmd in ['M', 'm']: ################################################
            [x, y] = [ float (c) for c in path_data[:2] ]
            del path_data[:2]

        elif cmd in ['Q', 'q'] : ###############################################
            [x1, y1, x, y] = [ float (c) for c in path_data[:4] ]
            del path_data[:4]

        elif cmd in ['S', 's'] : ###############################################
            [x2, y2, x, y] = [ float (c) for c in path_data[:4] ]
            del path_data[:4]

        elif cmd in ['T', 't'] : ###############################################
            [x, y] = [ float (c) for c in path_data[:2] ]
            del path_data[:2]

        elif cmd in ['V', 'v'] : ###############################################
            y = float( path_data[:1] )
            del path_data[:1]
            x = pts[-1][0]
            
        elif cmd in ['Z', 'z'] : ###############################################
            cmd = 'Z' # 
            [x, y] = [ pts[0][0], pts[0][1] ]

        else : #################################################################
            print 'this is not supposed to happen!'

        # fixing relative-absolute values
        if len(pts) > 0 and cmd.islower():
            x += pts[-1][0]
            y += pts[-1][1]

        pts.append( [x,y] )

    return np.array(pts)


################################################################################

def svg_to_ymal(svg_file_name):
    '''
    note: using np.roll, would connect the first and last points!
    for (p1, p2) in zip( pts, np.roll(pts, 1, axis=0) ):
        trait_dict['segments'].append( [p1[0], p1[1], p2[0], p2[1]] )
    '''

    yaml_file_name = svg_file_name.split('.')[0]+'.yaml'
    trait_dict = { 'lines': [], #[x1,y1,x2,y2]
                   'segments': [], #[x1,y1,x2,y2]
                   'rays': [], #[x1,y1,x2,y2]
                   'circles': [], #[cx,cy,cr]
                   'arcs': [] } #[cx,cy,cr,t1,t2]

    tree = ET.parse( svg_file_name )
    elements_dict = xml_tree_parser_to_svg_elements(tree)

    pts_lst = []
    ### path_element
    for path_elmt in elements_dict['path']:
        # NOTE: todo: assuming path only contains straight lines
        pts = svg_parser_path_element(path_elmt)
        pts_lst.append(pts)

    ### polygon_element
    for polygon_elmt in elements_dict['polygon']:
        pts = svg_parser_polygon_element(polygon_elmt)
        pts_lst.append(pts)

    ### polyline_element
    for polyline_elmt in elements_dict['polyline']:
        pts = svg_parser_polyline_element(polyline_elmt)
        pts_lst.append(pts)

    ### rect_element
    for rect_elmt in elements_dict['rect']:
        [x,y, w,h] = svg_parser_rect_element(rect_elmt)
        pts = [ [x, y], [x+w, y], [x+y, y+h], [x, y+h], [x, y] ]
        pts_lst.append(np.array(pts))

    ### line_element
    for line_elmt in elements_dict['line']:
        [x1,y1, x2,y2] = svg_parser_line_element(line_elmt)
        pts = [ [x1, y1], [x2, y2] ]
        pts_lst.append(np.array(pts))

    ### pts_lst is a list of pts
    # each pts contains a sequence of points in a path, polygone,...
    for pts in pts_lst:
        trait_dict['segments'] += [ list([ float(pts[idx,  0]),
                                           float(pts[idx,  1]),
                                           float(pts[idx+1,0]),
                                           float(pts[idx+1,1]) ])
                                    for idx in range(len(pts)-1) ]

    ### circle_element
    for circle_elmt in elements_dict['circle']:
        trait_dict['circles'].append( [float(n)
                                       for n in list(svg_parser_circle_element(circle_elmt))] )

    ### deleting empty fields in the dictionaty
    for k in trait_dict.keys():
        if len(trait_dict[k])==0:
            trait_dict.pop(k)
   
    ### saving data
    with open(yaml_file_name, 'w') as yaml_file:
            yaml.dump(trait_dict, yaml_file) #, default_flow_style=True)

    
    return yaml_file_name


################################################################################
###################################################################### deploying
################################################################################

if 0: 
    file_name = 'circle_lines.svg'
    # file_name = 'intel-01-occ-05cm.svg'
    # file_name = 'svg_test_case_complete.svg'
    # file_name = 'svg_test_case.svg'
    # file_name = 'rect_circ_poly_line.svg'
    # file_name = 'long_straight_path.svg'
    
    breaking_line = '\n------------------------'
    print 3 * breaking_line
        
    yaml_file_name = svg_to_ymal(file_name)
