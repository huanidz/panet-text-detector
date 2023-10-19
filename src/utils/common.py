from shapely.geometry import Polygon
import pyclipper

def count_parameters(model, count_non_trainable=False):
    if count_non_trainable:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculating_scaling_offset(poly_coordinate, r:float):
    """
    Get the offset (or margin) between original polygon and target offset polygon
    
    For inwardly offseting, expect r: to be < 1.0 (and bigger than 0.0 of course)
    
    Ideas from paper: "Shape Robust Text Detection with Progressive Scale Expansion Network" - Wenhai et. al
    """
    polygon = Polygon(poly_coordinate)
    area = polygon.area
    perimeter = polygon.length
    offset = area * (1.0 - r**2.0) / perimeter
    return offset    
    
    
def offset_polygon(polygon, offset):
    """
    Offset a polygon inwardly by the given offset.

    Args:
        polygon (list[tuple]): A list of 2D points representing the polygon.
        offset (float): The amount of inward offset for the polygon.

    Returns:
        list[list[tuple]]: A list of lists of 2D points representing the offset polygon(s).

    Example:
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        offset_square = offset_polygon(square, 2)
        print(offset_square)
        # Output: [[(2, 2), (8, 2), (8, 8), (2, 8)]]

    Note:
        - The function uses the Pyclipper library to perform the offset operation.
        - A negative offset value should be used to create an inward contraction effect.
        - The function returns a list of polygons, even if there's just one offset polygon.
    """
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(polygon, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    return pco.Execute(-offset)