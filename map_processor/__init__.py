from skimage import io
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def to_edges(map):
    if isinstance(map, str):
        map = io.imread(map)
    bitmask = np.any(map != 0, axis=2).astype(np.byte)
    edges = convolve2d(bitmask, np.array([
        [1, 1],
        [1, 1]
    ]))
    edges = np.all([edges != 4, edges != 0], axis=0).astype(np.byte)
    return edges


def edges_img_to_vertices(edges):
    working_edges = np.copy(edges)
    lines = []

    def start_line(start, start_dir):
        line = []
        to_start = [(start, start_dir)]
        while len(to_start) != 0:
            v0, dir = to_start.pop()
            i, j = v0
            while working_edges[i + dir[0], j + dir[1]] == 1:
                i, j = i + dir[0], j + dir[1]
                working_edges[i, j] = 0
            v1 = i, j
            line.append((v0, v1))

            # Search if line can be continued in orthogonal directions
            if working_edges[i + dir[1], j - dir[0]] == 1:
                to_start.append((v1, (dir[1], -dir[0])))
            elif working_edges[i - dir[1], j + dir[0]] == 1:
                to_start.append((v1, (-dir[1], dir[0])))
        working_edges[start] = 0
        lines.append(line)

    while (s := np.argmax(working_edges)) != 0:
        i, j = (s // working_edges.shape[0], s % working_edges.shape[1])
        if working_edges[i, j + 1] == 1:
            start_line((i, j), (0, 1))
        elif working_edges[i + 1, j] == 1:
            start_line((i, j), (1, 0))
        else:
            working_edges[i, j] = 0

    return lines


def cut_line_corners(line, min_dist=2):
    if len(line) < 2:
        return line
    out = list(line)
    for i in range(len(line)):
        edge1 = out[i]
        edge2 = out[(i + 1) % len(out)]
        if edge1 is None or edge2 is None or edge1[1] != edge2[0]:
            continue
        d1 = ((edge1[0][0] - edge1[1][0])**2 + (edge1[0][1] - edge1[1][1])**2)**0.5
        d2 = ((edge2[0][0] - edge2[1][0])**2 + (edge2[0][1] - edge2[1][1])**2)**0.5
        if d1 + d2 <= min_dist:
            out[i] = None
            out[(i + 1) % len(out)] = (edge1[0], edge2[1])
    return list(filter(lambda x: x is not None, out))


def cut_corners(lines, min_dist=2):
    return list(map(lambda line: cut_line_corners(line, min_dist), lines))


def plot_lines(lines):
    for line in lines:
        for v in line:
            plt.plot([v[0][1], v[1][1]], [-v[0][0], -v[1][0]], '.-r', markersize=5)
    plt.show()

