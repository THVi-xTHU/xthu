import numpy as np

def overlap_ratio(ex_box, gt_box):
    paded_gt = np.tile(gt_box, [ex_box.shape[0],1])
    insec = intersection(ex_box, paded_gt)
    uni = union(ex_box, paded_gt) - insec
    return insec / uni

def union(a, b):
    return (a[:, 2] - a[:,0]) * (a[:, 3] - a[:,1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:,1])


def intersection(a, b):
    x = np.maximum(a[:, 0], b[:, 0])
    y = np.maximum(a[:, 1], b[:, 1])
    w = np.minimum(a[:, 2], b[:, 2]) - x
    h = np.minimum(a[:, 3], b[:, 3]) - y
    return np.maximum(w, 0) * np.maximum(h, 0)


def bbox_overlaps(bbox1, bbox2):
    rows = []
    for bbox in bbox1:
        row = overlap_ratio(bbox2, bbox)
        rows.append(row)
    return np.array(rows)