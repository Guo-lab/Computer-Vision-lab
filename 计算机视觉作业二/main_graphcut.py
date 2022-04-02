
import cv2 # '4.5.5'
import numpy as np # '1.19.5'
import maxflow # (1, 2, 13)

import matplotlib.pyplot as plt # '3.2.2'
from medpy import metric # '0.3.0'




#@ https://www.cse.iitb.ac.in/~meghshyam/seminar/SeminarReport.pdf

#####################################################################
########################## Graph Maker Class #######################
#####################################################################
class GraphMaker:

    foreground = 1
    background = 0
    #!
    #//segmented = 1
    
    default = 0.5
    MAXIMUM = 1000000000
    
    

    def __init__(self,filename):
        self.image = None
        self.graph = None                         #(pixels)
        self.segment_overlay = None
        self.mask = None            
        self.load_image(filename)                # function to load image
        self.background_seeds = []               # BG seeds
        self.foreground_seeds = []               # FG seeds
        self.background_average = np.array(3)
        self.foreground_average = np.array(3)
        self.nodes = []
        self.edges = []



    def load_image(self, filename):
        self.image = cv2.imread(filename)
        self.graph = None                       # Without creating graph
        
        #! RGB with shape (768, 1024, 3)
        #////self.segment_overlay = np.zeros(self.image.shape[:2])    # GREY
        self.segment_overlay = np.zeros_like(self.image)
        
        self.mask = None
        
        
        
        
        

    #* Add seeds into BG and FG modes
    def add_seed(self, x, y, type):
        if self.image is None:
            print('Please load an image before adding seeds.')
            
        if type == self.background:
            if not self.background_seeds.__contains__((x, y)):
                self.background_seeds.append((x, y))
                #//cv2.rectangle(self.seed_overlay, (x-1, y-1), (x+1, y+1), (0, 0, 255), -1)
                
        elif type == self.foreground:
            if not self.foreground_seeds.__contains__((x, y)):
                self.foreground_seeds.append((x, y))
                #//cv2.rectangle(self.seed_overlay, (x-1, y-1), (x+1, y+1), (0, 255, 0), -1)








    ########################## Create Graph #######################
    def create_graph(self):
        #* With seeds, create graph
        if len(self.background_seeds) == 0 or len(self.foreground_seeds) == 0:
            print("Please enter at least one foreground and background seed.")
            return
        
        print("Making graph")
        
        print("Finding foreground and background averages")
        self.find_averages()
        
        print("Populating nodes and edges")
        self.populate_graph()

    
    
    def find_averages(self):
        # Fill the graph
        # shape gets the height and width of the image
        #@ https://www.delftstack.com/api/numpy/python-numpy-numpy.shape-function/#:~:text=Python%20NumPy%20numpy.shape%20%28%29%20function%20finds%20the%20shape,is%20the%20input%20array%20to%20find%20the%20dimensions.
        self.graph = np.zeros((self.image.shape[0], self.image.shape[1]))        
        print(self.graph.shape)                 # (768, 1024)
        self.graph.fill(self.default)           # 初始化填充为0.5 (default = 0.5)
        
        self.background_average = np.zeros(3)
        self.foreground_average = np.zeros(3)

        #@ Get RGB average
        for coordinate in self.background_seeds:
            self.graph[coordinate[1] - 1, coordinate[0] - 1] = 0
            self.background_average += self.image[coordinate[1], coordinate[0]]
        self.background_average /= len(self.background_seeds)

        for coordinate in self.foreground_seeds:
            self.graph[coordinate[1] - 1, coordinate[0] - 1] = 1
            self.foreground_average += self.image[coordinate[1], coordinate[0]]
            #//print(self.foreground_average)  
            # like [224. 205. 178.]
        self.foreground_average /= len(self.foreground_seeds)



    def populate_graph(self):
        self.nodes = []
        self.edges = []
        # make all s and t connections for the graph
        #@ numpy.ndenumerate
        #@ Return an iterator yielding pairs of array coordinates and values
        #@ https://numpy.org/doc/stable/reference/generated/numpy.ndenumerate.html#numpy-ndenumerate
        for (y, x), value in np.ndenumerate(self.graph):
            # Background pixel 
            # node in-out
            if value == 0.0:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), self.MAXIMUM, 0))
            # Foreground node
            elif value == 1.0:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), 0, self.MAXIMUM))
            # Neither BG nor FG
            else:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), 0, 0))

        for (y, x), value in np.ndenumerate(self.graph):
            if y == self.graph.shape[0] - 1 or x == self.graph.shape[1] - 1:
                continue                                    # At the edge of an image
            my_index = self.get_node_num(x, y, self.image.shape)

            #@ Create Edge in the graph
            #@ Power(): https://appdividend.com/2022/01/19/numpy-power/#:~:text=Numpy%20power%20The%20numpy%20power%20%28%29%20is%20a,main%20arguments%3A%201%29%20The%20array%20of%20base%202%29.
            neighbor_index = self.get_node_num(x+1, y, self.image.shape)
            # g: Weight of the edge (RGB difference)
            # [  ] => [  ] => [  ] => ...
            #   |      |
            #   V      V
            # [  ]    ...
            g = 1 / (1 + np.sum(np.power(self.image[y, x] - self.image[y, x+1], 2)))
            #//print("g is " + str(g))
            self.edges.append((my_index, neighbor_index, g))

            neighbor_index = self.get_node_num(x, y+1, self.image.shape)
            g = 1 / (1 + np.sum(np.power(self.image[y, x] - self.image[y+1, x], 2)))
            self.edges.append((my_index, neighbor_index, g))


    #@ Only used by "populate_graph"
    #* num is like index_id
    @staticmethod
    def get_node_num(x, y, array_shape):
        return y * array_shape[1] + x







    # command M
    ########################## Cut Graph #######################
    #% https://www.cse.iitb.ac.in/~meghshyam/seminar/SeminarReport.pdf
    # After Creating graph, cut the graph
    def cut_graph(self):
        self.segment_overlay = np.zeros_like(self.segment_overlay)  # segment_overlay clear
        self.mask = np.zeros_like(self.image, dtype=bool)           # mask: white or black
        
        #@ http://pmneila.github.io/PyMaxflow/maxflow.html
        g = maxflow.Graph[float](len(self.nodes), len(self.edges))
        nodelist = g.add_nodes(len(self.nodes))

        # For each node
        # Add an edge ‘SOURCE->i’ with capacity and another edge ‘i->SINK’ with capacity .
        #@ add_tedge(self, int i, long cap_source, long cap_sink)
        for node in self.nodes:
            g.add_tedge(nodelist[node[0]], node[1], node[2])
        
        # Adds a bidirectional edge between nodes and with the weights
        #@ add_edge(self, int i, int j, long capacity, long rcapacity)
        for edge in self.edges:
            g.add_edge(edge[0], edge[1], edge[2], edge[2])

        # 对图片开始执行切割    
        #@ Returns the capacity of the minimum cut or, equivalently, the maximum flow of the graph.
        flow = g.maxflow()
        print("maximum flow is {}".format(flow))

        for index in range(len(self.nodes)):
            #@ get_segment(self, i)
            #@ Returns which segment the given node belongs to.
            if g.get_segment(index) == 1:
                #* Find where this node is
                xy = self.get_xy(index, self.image.shape) 
    
                #////self.segment_overlay[xy[1], xy[0]] = 1
                #////self.mask[xy[1], xy[0]] = (True, True, True)
                #@ https://convertingcolors.com/rgb-color-1_193_37.html?search=RGB(1,%20193,%2037)
                self.segment_overlay[xy[1], xy[0]] = (1, 193, 37) # Segment part
                self.mask[xy[1], xy[0]] = (True, True, True)      # 


    #@ Only used in the 'cut_graph'
    @staticmethod
    def get_xy(nodenum, array_shape):
        return (nodenum % array_shape[1]), (int(nodenum / array_shape[1]))











    def save_image(self, outfilename):
        if self.mask is None:
            print('Please segment the image before saving.')
            return
        print(outfilename)
        # print(self.image.name())
        to_save = np.zeros_like(self.image)                         # Black
        #np.copyto(to_save, self.image)
        
        #@ https://numpy.org/doc/stable/reference/generated/numpy.copyto.html
        #@ image => to_save
        #@ where: array_like of bool, optional
        #! where
        #@ A boolean array which is broadcasted to match the dimensions of dst, and selects elements to copy from src to dst wherever it contains the value True.
        #//np.copyto(to_save, self.image, where=self.mask)            # mask => white
        OK = np.zeros_like(self.image)
        OK.fill(255)
        np.copyto(to_save, OK, where=self.mask)
        
        cv2.imwrite(outfilename, to_save)
        





    '''
    def swap_overlay(self, overlay_num):
        self.current_overlay = overlay_num
    '''
        
 


def draw_line(event, x, y, flags, param):
    ClickOn = False
    if event == cv2.EVENT_LBUTTONDOWN:
        ClickOn = True
        WorkFlow.add_seed(x - 1, y - 1, mode)

    elif event == cv2.EVENT_LBUTTONUP:
        ClickOn = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if ClickOn:
            WorkFlow.add_seed(x - 1, y - 1, mode)


if __name__ == '__main__':
    # 完成main函数
    
    filename = "./data/img/14.jpg"
    #filename = "./data/img/7.jpg"
    #filename = "./data/img/16.jpg"
    WorkFlow = GraphMaker(filename)
    
    WorkFlow.load_image(filename)
    
    #WorkFlow.add_seed(1, 1, 0)
    # (768, 1024)
    #WorkFlow.add_seed(384, 512, 1)
    
    # Add seeds artificially    
    
    # Initialize
    global mode
    mode = WorkFlow.foreground
    Display = np.array(WorkFlow.image)
    Window = 'Graph Cut'
    
    cv2.namedWindow(Window)
    cv2.setMouseCallback(Window, draw_line)

    while True:
        #@ cv2.addWeighted(src1, alpha, src2, beta, y)
        #@ img = alpha(src1) + beta(src2) + y (beta = 1 - alpha)
        display = cv2.addWeighted(Display, 0.9, WorkFlow.segment_overlay, 0.1, 0)
        cv2.imshow(Window, display)
        
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        
        elif key == ord('g'):
            WorkFlow.create_graph()
            WorkFlow.cut_graph()
            
        elif key == ord('t'):
            mode = 1 - mode
            print(mode)
                  
    #//print(WorkFlow.background_seeds)
    #//print(WorkFlow.foreground_seeds)
    WorkFlow.save_image('./data/out/out14.png')
    #//WorkFlow.save_image('./data/out/out7.png')
    #//WorkFlow.save_image('./data/out/out7.png')
    cv2.destroyAllWindows()
    
    pass