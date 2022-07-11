import cv2
import numpy as np

def draw_bbox(img, coords, color=(255,0,0), format='xy'):
    """Draw 2D bounding box using the coordinates in xywh(x, y, width, height
    and xy ([x_min, y_min], [x_max, y_max]) format. 
    Args:
        img: Image read in numpy array
        coords: 2D bounding box coordinates
        color: Color of the bounding box
        format: Format of the coordinates. xywh or xy format.
    
    Returns:
        Numpy array
    """
    if format == "xywh":
        x1, y1, w, h = coords
        x2, y2 = x1+w, y1+h
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    else:
        x1, y1 = map(int, coords[0])
        x2, y2 = map(int, coords[1])
    return cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def draw_polygon(img, coords, color=(255,0,0)):
    """Draw polygon using given list of coordinates
    Args:
        img: Imager in numpy array format
        coords: List of coodinates in [[x1,y1], [x2,y2], [x3,y3], ...] format
        color: Color of the Polygon
    
    Returns:
        Numpy array
    """
    pts = np.array(coords, np.int32)
    pts = pts.reshape((-1,1,2))
    return cv2.polylines(img, [pts], True, color, 2)

def convert_bbox_coords(bbox, format='xy_to_xywh'):
    """Convert bounding box format between xy and xywh
    Args:
        bbox: Coordinates of the bounding box
        format: String that indicates the bounding box coversion format
    Returns:
        Coordinates in xy or xywh format
    """
    if format.lower() not in ['xy_to_xywh', 'xywh_to_xy']:
        raise ValueError('Only xy_to_xywh or xywh_to_xy string is support.')

    if format.lower() == 'xy_to_xywh':
        xmin, ymin = bbox[0]
        xmax, ymax = bbox[1]
        w, h = xmax-xmin, ymax-ymin
        xmin, ymin, w, h = map(int, [xmin, ymin, w, h])
        return [xmin, ymin, w, h]
    else:
        xmin, ymin, w, h = bbox
        return [[xmin, ymin], [xmin+w, ymin+h]]
