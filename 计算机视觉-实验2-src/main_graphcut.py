
import cv2
import numpy as np
import maxflow

import matplotlib.pyplot as plt
from medpy import metric

class GraphMaker:

    foreground = 1
    background = 0
    segmented = 1
    default = 0.5
    MAXIMUM = 1000000000

    def __init__(self,filename):
        self.image = None
        self.graph = None
        self.segment_overlay = None
        self.mask = None
        self.load_image(filename)
        self.background_seeds = []
        self.foreground_seeds = []
        self.background_average = np.array(3)
        self.foreground_average = np.array(3)
        self.nodes = []
        self.edges = []

    def load_image(self, filename):
        self.image = cv2.imread(filename)
        self.graph = None
        self.segment_overlay = np.zeros(self.image.shape[:2])
        self.mask = None

    def add_seed(self, x, y, type):
        if self.image is None:
            print('Please load an image before adding seeds.')
        if type == self.background:
            if not self.background_seeds.__contains__((x, y)):
                self.background_seeds.append((x, y))
        elif type == self.foreground:
            if not self.foreground_seeds.__contains__((x, y)):
                self.foreground_seeds.append((x, y))

    def create_graph(self):
        if len(self.background_seeds) == 0 or len(self.foreground_seeds) == 0:
            print("Please enter at least one foreground and background seed.")
            return

        print("Making graph")
        print("Finding foreground and background averages")
        self.find_averages()

        print("Populating nodes and edges")
        self.populate_graph()


    def find_averages(self):
        self.graph = np.zeros((self.image.shape[0], self.image.shape[1]))
        print(self.graph.shape)
        self.graph.fill(self.default) # 初始化填充为0.5
        self.background_average = np.zeros(3)
        self.foreground_average = np.zeros(3)

        for coordinate in self.background_seeds:
            self.graph[coordinate[1] - 1, coordinate[0] - 1] = 0
            self.background_average += self.image[coordinate[1], coordinate[0]]

        self.background_average /= len(self.background_seeds)

        for coordinate in self.foreground_seeds:
            self.graph[coordinate[1] - 1, coordinate[0] - 1] = 1
            self.foreground_average += self.image[coordinate[1], coordinate[0]]

        self.foreground_average /= len(self.foreground_seeds)

    def populate_graph(self):
        self.nodes = []
        self.edges = []
        for (y, x), value in np.ndenumerate(self.graph):
            if value == 0.0:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), self.MAXIMUM, 0))

            elif value == 1.0:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), 0, self.MAXIMUM))

            else:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), 0, 0))

        for (y, x), value in np.ndenumerate(self.graph):
            if y == self.graph.shape[0] - 1 or x == self.graph.shape[1] - 1:
                continue
            my_index = self.get_node_num(x, y, self.image.shape)

            neighbor_index = self.get_node_num(x+1, y, self.image.shape)
            g = 1 / (1 + np.sum(np.power(self.image[y, x] - self.image[y, x+1], 2)))
            # print("g is " + str(g))
            self.edges.append((my_index, neighbor_index, g))

            neighbor_index = self.get_node_num(x, y+1, self.image.shape)
            g = 1 / (1 + np.sum(np.power(self.image[y, x] - self.image[y+1, x], 2)))
            self.edges.append((my_index, neighbor_index, g))

    def cut_graph(self):
        self.segment_overlay = np.zeros_like(self.segment_overlay)
        self.mask = np.zeros_like(self.image, dtype=bool)
        g = maxflow.Graph[float](len(self.nodes), len(self.edges))
        nodelist = g.add_nodes(len(self.nodes))


        for node in self.nodes:
            g.add_tedge(nodelist[node[0]], node[1], node[2])

        for edge in self.edges:
            g.add_edge(edge[0], edge[1], edge[2], edge[2])

        flow = g.maxflow()
        print("maximum flow is {}".format(flow))

        for index in range(len(self.nodes)):
            if g.get_segment(index) == 1:
                xy = self.get_xy(index, self.image.shape)
                self.segment_overlay[xy[1], xy[0]] = 1
                self.mask[xy[1], xy[0]] = (True, True, True)

    def swap_overlay(self, overlay_num):
        self.current_overlay = overlay_num

    def save_image(self, outfilename):
        if self.mask is None:
            print('Please segment the image before saving.')
            return
        print(outfilename)
        # print(self.image.name())
        to_save = np.zeros_like(self.image)

        np.copyto(to_save, self.image, where=self.mask)
        cv2.imwrite(outfilename, to_save)
        return self.segment_overlay

    @staticmethod
    def get_node_num(x, y, array_shape):
        return y * array_shape[1] + x

    @staticmethod
    def get_xy(nodenum, array_shape):
        return (nodenum % array_shape[1]), (int(nodenum / array_shape[1]))

if __name__ == '__main__':
    # 完成main函数

    pass


