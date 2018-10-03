
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import Tkinter, Tkconstants, tkFileDialog
from Tkinter import PhotoImage, Canvas, Frame, BOTH, NW
import os
# from fibHeap import FibonacciHeap
import cv2
from PIL import Image, ImageTk
import numpy as np
import copy
from fibonacci_heap_mod import Entry, Fibonacci_heap
import csv
import matplotlib.animation as animation
import time
from heapq import heappush, heappop
import heapq
from heapdict import heapdict
import itertools
from scipy import signal
import math

class Scissor:
    def __init__(self, master):
        # Create a container
        frame = Tkinter.Frame(master)
        self.root = master
        self.curPath = os.getcwd()
        # Create 2 buttons
        self.button_img_open = Tkinter.Button(frame,text="open file",
                                        command=self.fileOpen)
        self.button_img_open.pack(side="left")
        #
        self.button_save_contour = Tkinter.Button(frame,text="save_contour",
                                        command=self.save_contour)
        self.button_save_contour.pack(side="left")

        self.button_loadContour = Tkinter.Button(frame, text="load contour",
                                                 command=self.loadContour)
        self.button_loadContour.pack(side="left")

        self.button_img_only = Tkinter.Button(frame,text="Hide Contour",
                                        command=self.hidden_contour)
        self.button_img_only.pack(side="left")

        self.button_img_only = Tkinter.Button(frame,text="show contour drawn before",
                                        command=self.show_contour)
        self.button_img_only.pack(side="left")

        self.button_draw_contour = Tkinter.Button(frame,text="draw contour",
                                        command=self.draw_contour)
        self.button_draw_contour.pack(side="left")

        self.button_stop_contour = Tkinter.Button(frame,text="stop drawing",
                                        command=self.stop_contour)
        self.button_stop_contour.pack(side="left")

        self.button_remove_old_contour = Tkinter.Button(frame,text="remove old contour",
                                        command=self.remove_old_contour)
        self.button_remove_old_contour.pack(side="left")
        #
        self.button_cropImage = Tkinter.Button(frame,text="cropImage",
                                        command=self.cropImage)
        self.button_cropImage.pack(side="left")
        #
        self.button_saveCropped = Tkinter.Button(frame, text="Save Cropped Image",
                                               command=self.saveCropped)
        self.button_saveCropped.pack(side="left")



        # self.button_scaleDown = Tkinter.Button(frame, text="Scale Down Image",
        #                                          command=self.saveCropped)
        # self.button_scaleDown.pack(side="left")
        #

        #
        # self.button_pathTree = Tkinter.Button(frame, text="Path Tree",
        #                                        command=self.pathTree)
        # self.button_pathTree.pack(side="left")
        # #
        # self.button_minPath = Tkinter.Button(frame, text="Minimal Path",
        #                                        command=self.minPath)
        # self.button_minPath.pack(side="left")

        self.imgPath = None
        self.imgFileName = None
        self.img = None

        self.contour = []
        self.imgTk = None
        self.imgPIL = None
        self.grayImgNp = None
        self.colorImgNp = None
        self.croppedImage = None

        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(131)
        self.ax1.set_axis_off()
        # pixelNode
        self.ax2 = self.fig1.add_subplot(132)
        self.ax2.set_axis_off()
        # CostGraph
        self.ax3 = self.fig1.add_subplot(133)
        self.ax3.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.fig1,master=master)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.mouseEvent = self.canvas.mpl_connect('motion_notify_event', self.get_cursor_position)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        self.startDrawContour = False
        self.shownContour = False
        self.firstpt = True
        self.currentPath = []
        self.lastClickPt = None
        self.savedClicked = []
        self.pixelGraphTemplate = None
        self.pixelGraphSP = None
        self.ann_cross = []
        self.ann_line = []
        self.gettingPath = False
        self.testPt = []
        self.temp_line = []
        self.colorbar = []
        self.cropped = False
        frame.pack()

    def reInit(self):
        self.cropped = False
        self.startDrawContour = False
        self.shownContour = False
        self.firstpt = True
        self.currentPath = []
        self.lastClickPt = None
        self.savedClicked = []
        self.pixelGraphTemplate = None
        self.pixelGraphSP = None
        self.ann_cross = []
        self.ann_line = []
        self.gettingPath = False
        self.testPt = []
        self.temp_line = []
        for bar in self.colorbar:
            bar.remove()
        self.colorbar = []
        # self.ax2.clear()
        # self.ax3.clear()

    def closeContour(self):
        if self.contour[0][0] != self.contour[-1][0] or self.contour[0][1] != self.contour[-1][1]:
            if len(self.temp_line) > 0:
                self.temp_line[-1].remove()
            else:
                print len(self.temp_line)
            self.pixelGraphSP = self.Pixelgraph(self.grayImgNp, self.colorImgNp)
            self.pixelGraphSP.buildMST(self.contour[-1][0], self.contour[-1][1])
            y, x = self.contour[0][1], self.contour[0][0]
            self.currentPath = self.pixelGraphSP.findShortestPath(x, y)
            self.contour = self.contour + self.currentPath
            self.ax1.hold('on')
            line, = self.ax1.plot([pt[1] for pt in self.currentPath], [pt[0] for pt in self.currentPath], color='blue')
            self.temp_line.append(line)
            self.canvas.draw()

    def cropImage(self):
        if self.cropped is False:
            self.closeContour()
            mask = np.zeros(self.grayImgNp.shape)
            tempContour = np.array(self.contour)
            tempContour[:,[0,1]] = tempContour[:, [1,0]]
            cv2.drawContours(mask, [tempContour], 0, 255, -1)

            self.croppedImage = np.ones(self.colorImgNp.shape)*255
            self.croppedImage[mask==255,:] = self.colorImgNp[mask==255,:]
            self.croppedImage = self.croppedImage.astype('uint8')

            self.ax1.clear()
            self.ax1.imshow(self.croppedImage)
            self.ax1.set_axis_off()
            self.temp_line = []
            self.canvas.draw()
            self.cropped = True

    def saveCropped(self):
        self.cropImage()
        imgFilenameList = self.imgFileName.split('.')
        initialfile = imgFilenameList[0].split('/')[-1] + '_cropped.'+imgFilenameList[-1]
        filename = tkFileDialog.asksaveasfilename(initialdir=self.imgPath, initialfile=initialfile, title="Save file")
        cv2.imwrite(filename, cv2.cvtColor(self.croppedImage, cv2.COLOR_BGR2RGB))

    # def readtestPt(self):
    #     csvFile = 'test.csv'
    #     self.testPt = []
    #     with open(csvFile, 'r') as csvin:
    #         csvreader = csv.reader(csvin, delimiter=',')
    #         for line in csvreader:
    #             self.testPt.append((int(line[0]), int(line[1])))
    def loadContour(self):
        if self.imgPath is not None:
            filename = tkFileDialog.askopenfilename(initialdir=self.imgPath, title="Select Contour File", filetypes = (("csv files","*.csv"),("all files","*.*")))
            self.contour = np.loadtxt(filename)
            self.ax1.plot([pt[1] for pt in self.contour], [pt[0] for pt in self.contour], color='blue')

    def pathTree(self):
        print 'pathTree'

    def minPath(self):
        print 'minPath'

    def draw_contour(self):
        self.startDrawContour = True

    def stop_contour(self):
        print 'stop drawing'
        self.startDrawContour = False

    def hidden_contour(self):
        print 'hidden contour'
        if self.startDrawContour is False:
            if self.shownContour is True:
                for cross in self.ann_cross:
                    cross.set_visible(not cross.get_visible())
                for line in self.ann_line:
                    line.set_visible(not line.get_visible())
                self.canvas.draw()
                self.shownContour = False

    def show_contour(self):
        if self.shownContour is False:
            for cross in self.ann_cross:
                cross.set_visible(not cross.get_visible())
            for line in self.ann_line:
                line.set_visible(not line.get_visible())
            self.canvas.draw()
            self.shownContour = True

        self.canvas.draw()

    def remove_old_contour(self):
        self.contour = []
        self.startDrawContour = False
        self.currentPath = []
        self.savedClicked = []
        self.lastClickPt = None
        self.pixelGraphSP = None
        self.shownContour = False
        for cross in self.ann_cross:
            cross.remove()
        for line in self.ann_line:
            line.remove()
        self.canvas.draw()
        
    def on_click(self, event):
        if event.dblclick:
            # end drawing
            print 'end drawing'
            if event.inaxes is not None:
                self.startDrawContour = False
        else:
            if event.inaxes is not None:
                if self.startDrawContour:
                    self.shownContour = True
                    print 'start drawing'
                    y, x = int(event.xdata), int(event.ydata)
                    print x, y
                    for bar in self.colorbar:
                        bar.remove()
                    self.colorbar = []
                    self.pixelGraphSP = self.Pixelgraph(self.grayImgNp, self.colorImgNp)
                    self.pixelGraphSP.buildMST(x,y)
                    # print self.contour
                    # print self.currentPath
                    self.contour = self.contour + self.currentPath
                    self.currentPath = []
                    cross, = self.ax1.plot(y, x, '+', color='red')
                    costMap = self.pixelGraphSP.costMap()
                    im = self.ax3.imshow(costMap)
                    bar = self.fig1.colorbar(im)

                    self.ax2.clear()
                    neighborMap = self.pixelGraphSP.nodeNeighbor(x, y)
                    self.ax2.imshow(neighborMap)
                    self.ax2.set_title('Node and cost')
                    self.ax2.set_axis_off()
                    # self.ax2.grid(True)
                    # self.ax2.grid(which='minor', color='black', linestyle='-', linewidth=2)

                    self.colorbar.append(bar)
                    self.ax3.set_title('Cost to each pixel')
                    self.ax3.set_axis_off()
                    self.canvas.draw()
                    self.lastClickPt = (x, y)
                    self.ann_cross.append(cross)
                    self.savedClicked.append([y, x])
                    self.ann_line.append(self.temp_line[-1])
                    self.temp_line = []
                    self.canvas.draw()
                    # self.ann_cross.append(cross)
                    # self.ax2.imshow(self.pixelGraphSP.costMap())
                    # self.fig1.colorbar(im)
                    # self.canvas.draw()
            else:
                print 'Clicked ouside axes bounds but inside plot window'
    #
    # def update_plot(self):
    #     v = self.servo.getVelocity()
    #     t = self.servo.getTorque()
    #     self.add_point(self.velocity_line, v)
    #     self.add_point(self.torque_line, t)
    #     self.after(100, self.update_plot)

    # def draw_test(self):
    #     self.readtestPt()
    #     msttimeList = []
    #     pathtimeList = []
    #     for idx, pt in enumerate(self.testPt):
    #         nextpt = None
    #         if idx == (len(self.testPt) - 1):
    #             nextpt = self.testPt[0]
    #         else:
    #             nextpt = self.testPt[idx + 1]
    #         print idx, pt
    #         mstTime, pathTime = self.plotPt(pt, nextpt)
    #         msttimeList.append(mstTime)
    #         pathtimeList.append(pathTime)
    #     print 'MST time'
    #     print msttimeList
    #     print 'Path time'
    #     print pathtimeList


    # def plotPt(self, pt, nextpt):
    #     print 'start drawing'
    #     # if self.lastClickPt is None:
    #     self.pixelGraphSP = copy.deepcopy(self.pixelGraphTemplate)
    #     mstT0 = time.time()
    #     self.pixelGraphSP.buildMST(pt[0],pt[1])
    #     mstTime = time.time()-mstT0
    #     print self.currentPath
    #     pathT0 = time.time()
    #     self.currentPath = self.pixelGraphSP.findShortestPath(nextpt[0], nextpt[1])
    #     pathTime = time.time()-pathT0
    #     self.contour = self.contour + self.currentPath
    #     cross = self.ax1.plot(nextpt[1], nextpt[0], '+', c='red')
    #     line = self.ax1.plot([p[1] for p in self.currentPath], [p[0] for p in self.currentPath], c='blue')
    #     self.canvas.draw()
    #     time.sleep(10)
    #
    #     return mstTime, pathTime

    def get_cursor_position(self, event):
        if self.startDrawContour:
            if self.lastClickPt is not None:
                if event.inaxes is not None:
                    if self.gettingPath is False:
                        self.gettingPath = True
                        if len(self.temp_line) > 0:
                            self.temp_line[-1].remove()
                        else:
                            print len(self.temp_line)
                        print 'cursor:'
                        y, x = int(event.xdata), int(event.ydata)
                        self.currentPath = self.pixelGraphSP.findShortestPath(x, y)
                        self.ax1.hold('on')
                        line, = self.ax1.plot([pt[1] for pt in self.currentPath], [pt[0] for pt in self.currentPath], color='blue')
                        self.temp_line.append(line)

                        self.canvas.draw()
                        self.gettingPath = False
                else:
                    print 'Cursor ouside axes bounds but inside plot window'



    def pil2cv(self):
        self.grayImgNp = np.array(self.imgPIL.convert('L'))
        self.colorImgNp = np.array(self.imgPIL.convert('RGB'))


    def fileOpen(self):
        self.reInit()
        filename = tkFileDialog.askopenfilename(initialdir=self.curPath, title="Select file")
        print filename
        path, fname = os.path.split(os.path.abspath(filename))
        self.imgPath = path
        self.imgFileName = filename
        # print self.imgFileName
        self.imgPIL = Image.open(filename)
        self.imgTk = ImageTk.PhotoImage(self.imgPIL)
        self.pil2cv()
        # self.pixelGraphTemplate = self.Pixelgraph(self.grayImgNp, self.colorImgNp)
        # self.tkImg = PhotoImage(filename=filename)
        # print self.img
        self.show_img_only()
        self.showEdgeMap = False
    #
    # def tkimFromCv2(self, img):
    #     if len(img.shape) == 3 and img.shape[2] == 3:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = Image.fromarray(img)
    #     img = ImageTk.PhotoImage(img)
    #     self.tkImg = img
    # #
    # #
    # class pixel_Graph:
    # #     def __init__(self, img):
    # def LiveWireDP(self, x, y):
    #     pixelGraph = self.pixelGraph(self.grayImgNp)
    #
    #     self.canvas.draw()

    def inBoundary(self, x, y):
        size = self.grayImgNp.shape
        if x > size[0] or x < 0:
            return False
        elif y > size[1] or y < 0:
            return False
        return True

    def save_contour(self):
        initialfile = self.imgFileName.split('.')[0].split('/')[-1]+'_contour.csv'
        filename = tkFileDialog.asksaveasfilename(initialdir=self.imgPath, initialfile=initialfile, title="Save file")
        print filename
        np.savetxt(filename, self.contour, dtype=int)



    def show_img_only(self):
        # self.ax1.clear()
        # print self.img.shape
        # self.tkImg = self.tkimFromCv2(self.img)
        self.ax1.clear()
        # if self.grayImgNp.shape[0] * self.grayImgNp.shape[1] > (300*300):
        #     if self.grayImgNp.shape[1] > self.grayImgNp.shape[0]:
        #         self.grayImgNp = np.array(cv2.resize(self.grayImgNp, (300, self.grayImgNp.shape[1]*300/self.grayImgNp.shape[0])))
        #         self.colorImgNp = np.array(cv2.resize(self.colorImgNp,
        #                                    (300, self.grayImgNp.shape[1] * 300 / self.grayImgNp.shape[0])))
        #     else:
        #         self.grayImgNp = np.array(cv2.resize(self.grayImgNp,
        #                                               (self.grayImgNp.shape[0] * 300 / self.grayImgNp.shape[1], 300)))
        #         self.colorImgNp = np.array(cv2.resize(self.colorImgNp,
        #                                     (self.grayImgNp.shape[0] * 300 / self.grayImgNp.shape[1], 300)))
        self.ax1.imshow(self.colorImgNp, 'gray', interpolation="nearest")
        self.ax1.set_axis_off()
        self.canvas.draw()

        # self.canvas.create_image(50, 10, image=self.tkImg, anchor=NW)
        # self.canvas.draw()
        # self.canvas.create_image(20,20, anchor=NW, image=self.imgTk)

    class Pixelgraph:
        class pixelNode:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.neighbors = None
                # state: -1 = initial
                #         0 = expanded
                #         1 = active
                self.state = -1
                self.totalCost = 0
                self.prevNode = None
                self.drawn = False

        def __init__(self, grayimg, colorimg):
            self.grayimg = grayimg
            self.colorimg = colorimg
            self.pixelNodeList2D = {}
            self.startpt = None
            self.built_MST = False
            self.noNeighbor = 8
            edgeDetectorList = np.array([
                                            # (-1,-1)
                                            [[[0,1,0],
                                             [-1,0,0],
                                             [0,0,0]],
                                             # (-1,0)
                                             [[1,0,-1],
                                              [1,0,-1],
                                              [0,0,0]],
                                             # (-1,1)
                                             [[0,1,0],
                                              [0,0,-1],
                                              [0,0,0]]],
                                                # (0,-1)
                                            [[[1,1,0],
                                              [0,0,0],
                                              [-1,-1,0]],
                                            #  (0,0)
                                            [[1,1,1],
                                             [1,1,1],
                                             [1,1,1]],
                                            #  (0,1)
                                            [[0,1,1],
                                             [0,0,0],
                                             [0,-1,-1]]],
                                                # (1,-1)
                                            [[[0,0,0],
                                              [1,0,0],
                                              [0,-1,0]],
                                             # (1,0)
                                             [[0,0,0],
                                              [1,0,-1],
                                              [1,0,-1]],
                                             # (1,1)
                                             [[0,0,0],
                                              [0,0,1],
                                              [0,-1,0]]] ]).astype('float')

            # edgeDetectorListOrth = np.array([
            #                                 # (-1,-1)
            #                                 [[[1, 0, 0],
            #                                   [0, -1, 0],
            #                                   [0, 0, 0]],
            #                                  # (-1,0)
            #                                  [[1, 1, 1],
            #                                   [-1, -1, -1],
            #                                   [0, 0, 0]],
            #                                  # (-1,1)
            #                                  [[0, 0, 1],
            #                                   [0, -1, 0],
            #                                   [0, 0, 0]]],
            #                                 # (0,-1)
            #                                 [[[1, -1, 0],
            #                                   [1, -1, 0],
            #                                   [1, -1, 0]],
            #                                  #  (0,0)
            #                                  [[1, 1, 1],
            #                                   [1, 1, 1],
            #                                   [1, 1, 1]],
            #                                  #  (0,1)
            #                                  [[0, 1, -1],
            #                                   [0, 1, -1],
            #                                   [0, 1, -1]]],
            #                                 # (1,-1)
            #                                 [[[0, 0, 0],
            #                                   [0, -1, 0],
            #                                   [1, 0, 0]],
            #                                  # (1,0)
            #                                  [[0, 0, 0],
            #                                   [1, 1, 1],
            #                                   [-1,-1, -1]],
            #                                  # (1,1)
            #                                  [[0, 0, 0],
            #                                   [0, 1, 0],
            #                                   [0, 0, -1]]]]).astype('float')

            def normalizationOfFilter(filter):
                sumOfFilter = 0
                for r in filter:
                    for c in r:
                        sumOfFilter += abs(c)
                normalized =  filter*(1/sumOfFilter)
                # print normalized
                return normalized


            def multipleLength(filters):
                result = []
                for r in range(3):
                    tempEdgeList = []
                    for c in range(3):
                        if abs(r-c) % 2 == 0:
                            tempEdgeList.append(normalizationOfFilter(filters[r, c])*np.sqrt(2))
                        else:
                            tempEdgeList.append(normalizationOfFilter(filters[r, c]))
                    result.append(tempEdgeList)
                return np.array(result)

            self.edgeDetector = multipleLength(normalizationOfFilter(edgeDetectorList))
            # self.edgeDetectorOrth = multipleLength(normalizationOfFilter(edgeDetectorListOrth))

            self.edgeMap = []
            for r in range(3):
                tempEdgeList = []
                for c in range(3):
                    edgeCost = np.zeros(self.grayimg.shape)
                    for color in range(3):
                        convEdge = np.abs(signal.convolve2d(colorimg[:,:,color], self.edgeDetector[r,c], boundary='symm', mode='same'))
                        edgeCost += (max(convEdge.ravel()) - convEdge)
                        # convEdgeOrth = np.abs(signal.convolve2d(colorimg[:,:,color], self.edgeDetectorOrth[r,c], boundary='symm', mode='same'))
                        # edgeCost += convEdgeOrth
                    tempEdgeList.append(edgeCost/3)
                self.edgeMap.append(tempEdgeList)

            self.edgeMap = np.array(self.edgeMap)
            # print self.edgeMap.shape

            for i in range(grayimg.shape[0]):
                for j in range(grayimg.shape[1]):
                    pixelNode = self.pixelNode(i, j)
                    pixelNode.neighbors = self.buildNeighbor(i,j)
                    self.pixelNodeList2D[(i, j)] = pixelNode

        def buildNeighbor(self, i, j):
            neighbor = {}
            for r in range(3):
                x = i + r - 1
                for c in range(3):
                    y = j + c - 1
                    if not(r == 1 and c == 1):
                        if self.inBoundary(x, y):
                            neighbor[(x,y)] = self.edgeMap[r,c, i, j]
            return neighbor
        # using heapdict
        # def buildMST(self, startx, starty):
        #     print 'building MST'
        #     mstT0 = time.time()
        #     heap = heapdict()
        #     self.startpt = (startx, starty)
        #     # entry_finder = {}
        #     # REMOVED = '<removed-task>'
        #     # counter = itertools.count()
        #     #
        #     # def remove_task(task):
        #     #     'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        #     #     entry = entry_finder.pop(task)
        #     #     entry[-1] = REMOVED
        #     #
        #     # def add_task(task, priority=0):
        #     #     'Add a new task or update the priority of an existing task'
        #     #     if task in entry_finder:
        #     #         remove_task(task)
        #     #     count = next(counter)
        #     #     entry = [priority, count, task]
        #     #     entry_finder[task] = entry
        #     #     heappush(pq, entry)
        #     for neighCoor, neighCost in self.pixelNodeList2D[self.startpt].neighbors.iteritems():
        #         heap[neighCoor] = neighCost
        #         self.pixelNodeList2D[neighCoor].totalCost = neighCost
        #         self.pixelNodeList2D[neighCoor].prevNode = self.startpt
        #
        #     while len(heap) > 0:
        #         # print 'deq'
        #         # deqt0 = time.time()
        #         qCoor, qCost = heap.popitem()
        #         # mark q as EXPANDED (state = 0)
        #         self.pixelNodeList2D[qCoor].state = 0
        #         self.pixelNodeList2D[qCoor].totalCost = qCost
        #         for rCoor, rCost in self.pixelNodeList2D[qCoor].neighbors.iteritems():
        #             rState = self.pixelNodeList2D[rCoor].state
        #             if rState != 0:
        #                 if rState == -1:
        #                     self.pixelNodeList2D[rCoor].prevNode = qCoor
        #                     self.pixelNodeList2D[rCoor].totalCost = qCost + rCost
        #                     heap[rCoor] = qCost + rCost
        #                     self.pixelNodeList2D[rCoor].state = 1
        #                 else:
        #                     if self.pixelNodeList2D[rCoor].totalCost > (self.pixelNodeList2D[qCoor].totalCost + rCost):
        #                         self.pixelNodeList2D[rCoor].prevNode = qCoor
        #                         self.pixelNodeList2D[rCoor].totalCost = qCost + rCost
        #                         heap[rCoor] = qCost + rCost
        #         # deqTime = time.time() - deqt0
        #         # print deqTime
        #     self.pixelNodeList2D[self.startpt].prevNode = None
        #     mstTime = time.time() - mstT0
        #     print mstTime
        #     self.built_MST = True

        def buildMST(self, startx, starty):
            print 'building MST'
            mstT0 = time.time()
            heap = Fibonacci_heap()
            self.startpt = (startx, starty)
            heapTempGraph = {}
            for neighCoor, neighCost in self.pixelNodeList2D[self.startpt].neighbors.iteritems():
                heapTempGraph[neighCoor] = heap.enqueue(value=neighCoor, priority=neighCost)
                self.pixelNodeList2D[neighCoor].totalCost = neighCost
                self.pixelNodeList2D[neighCoor].prevNode = self.startpt

            while heap.m_size != 0:
                # print 'deq'
                # deqt0 = time.time()
                q = heap.dequeue_min()
                # mark q as EXPANDED (state = 0)
                self.pixelNodeList2D[q.m_elem].state = 0
                self.pixelNodeList2D[q.m_elem].totalCost = q.m_priority
                for rCoor, rCost in self.pixelNodeList2D[q.m_elem].neighbors.iteritems():
                    rState = self.pixelNodeList2D[rCoor].state
                    if rState != 0:
                        if rState == -1:
                            self.pixelNodeList2D[rCoor].prevNode = q.m_elem
                            self.pixelNodeList2D[rCoor].totalCost = self.pixelNodeList2D[q.m_elem].totalCost + rCost
                            heapTempGraph[rCoor] = heap.enqueue(value=rCoor, priority=self.pixelNodeList2D[rCoor].totalCost)
                            self.pixelNodeList2D[rCoor].state = 1
                        else:
                            if self.pixelNodeList2D[rCoor].totalCost > self.pixelNodeList2D[q.m_elem].totalCost + rCost:
                                self.pixelNodeList2D[rCoor].prevNode = q.m_elem
                                self.pixelNodeList2D[rCoor].totalCost = self.pixelNodeList2D[q.m_elem].totalCost + rCost
                                heap.decrease_key(heapTempGraph[rCoor], self.pixelNodeList2D[rCoor].totalCost)
                # deqTime = time.time() - deqt0
                # print deqTime
            self.pixelNodeList2D[self.startpt].prevNode = None
            mstTime = time.time() - mstT0
            print mstTime
            self.built_MST = True

        # def buildMST(self, startx, starty):
        #     print 'building MST'
        #     mstT0 = time.time()
        #     heap = FibonacciHeap()
        #     self.startpt = (startx, starty)
        #     for neighCoor, neighCost in self.pixelNodeList2D[self.startpt].neighbors.iteritems():
        #         print neighCoor, neighCost
        #         heap.insert(key=neighCost, value=neighCoor)
        #         self.pixelNodeList2D[neighCoor].totalCost = neighCost
        #         self.pixelNodeList2D[neighCoor].prevNode = self.startpt
        #
        #
        #     while heap is not None:
        #         q = heap.extract_min()
        #         # mark q as EXPANDED (state = 0)
        #         self.pixelNodeList2D[q.value].state = 0
        #         self.pixelNodeList2D[q.value].totalCost = q.key
        #         print 'q:'
        #         print q.value, q.key
        #         for rCoor, rCost in self.pixelNodeList2D[q.value].neighbors.iteritems():
        #             rState = self.pixelNodeList2D[rCoor].state
        #             print 'r'
        #             print rCoor, rCost
        #             if rState != 0:
        #                 if rState == -1:
        #                     self.pixelNodeList2D[rCoor].prevNode = q.value
        #                     self.pixelNodeList2D[rCoor].totalCost = self.pixelNodeList2D[q.value].totalCost + rCost
        #                     heap.insert(key=self.pixelNodeList2D[rCoor].totalCost, value=rCoor)
        #                     self.pixelNodeList2D[rCoor].state = 1
        #                 else:
        #                     if self.pixelNodeList2D[rCoor].totalCost > self.pixelNodeList2D[q.value].totalCost + rCost:
        #                         self.pixelNodeList2D[rCoor].prevNode = q.value
        #                         self.pixelNodeList2D[rCoor].totalCost = self.pixelNodeList2D[q.value].totalCost + rCost
        #                         heap.updateByValue(rCoor, self.pixelNodeList2D[rCoor].totalCost)
        #     mstTime = time.time() - mstT0
        #     print mstTime
        #     self.built_MST = True

        def findShortestPath(self, endx, endy):
            print 'Time for Shortest Path:'
            t0 = time.time()
            if self.built_MST:
                endpt = (endx, endy)
                nextpt = endpt
                pathTemp = []
                numOfpt = 0
                while (nextpt[0] != self.startpt[0] or nextpt[1] != self.startpt[1]):
                    nextpt = self.pixelNodeList2D[nextpt].prevNode
                    pathTemp.append(nextpt)
                    numOfpt += 1
                    if nextpt is None:
                        break
                path = []
                while len(pathTemp) != 0:
                    path.append(pathTemp.pop())
                timeUsed = time.time() - t0
                print timeUsed
                return path

        def costMap(self):
            costImg = np.zeros(self.grayimg.shape)
            for i in range(self.grayimg.shape[0]):
                for j in range(self.grayimg.shape[1]):
                    costImg[i,j] = self.pixelNodeList2D[(i,j)].totalCost
            return costImg
        #
        # def neighborNode(self, x, y):
        #     print 'test'
        #     neighborImg = np.zeros((9,9,3))
        #     for i in range(9):
        #         xq = i / 3 - 1
        #         for j in range(9):
        #             yq = j / 3 - 1
        #             if i % 3 == 1 and j % 3 == 1:
        #                 neighborImg[i,j] = self.colorimg[x+xq,y+yq,:]
        #             else:
        #                 xr = i % 3 - 1
        #                 yr = j % 3 - 1
        #                 edge = self.pixelNodeList2D[(x+xq,y+yq)].neighbors[(x+xr, y+yr)]
        #                 neighborImg[i,j] = np.repeat(edge, 3)
        #     return neighborImg

        def nodeNeighbor(self,x,y):
            neighborImg = np.zeros((9,9,3))
            for i in range(9):
                for j in range(9):
                    if i%3 == 1 and j%3 == 1:
                        neighborImg[i,j] = self.colorimg[(x-(i/3)-1),(y-(j/3)-1),:]
                    else:
                        neighborImg[i,j] = (self.edgeMap[i%3,j%3,(x-(i/3)-1),(y-(j/3)-1)]*255/max(self.edgeMap[i%3,j%3].ravel())*np.ones((3,)))
            return neighborImg.astype('uint8')


        def inBoundary(self, x, y):
            size = self.grayimg.shape
            if x >= size[0] or x < 0:
                return False
            elif y >= size[1] or y < 0:
                return False
            return True

        # def computePathCost(self, x, y):
        #     # neighImg = self.grayimg[(x - 1):(x + 1), (y - 1):(y + 1)]
        #     # print self.grayimg.shape
        #     costList = {}
        #     maxCost = 0
        #     for r in range(3):
        #         xcoor = x + r - 1
        #         for c in range(3):
        #             ycoor = y + c - 1
        #             # print 'coor'
        #             # print xcoor, ycoor
        #             # print 'row, col'
        #             # print r, c
        #             if self.inBoundary(xcoor, ycoor):
        #                 try:
        #                     if r == 0 and c == 0:
        #                         cost = np.abs(self.grayimg[xcoor, (ycoor+1)] - self.grayimg[(xcoor+1), ycoor]) / np.sqrt(2)
        #                     elif r == 0 and c == 1:
        #                         cost = np.abs(
        #                             self.grayimg[xcoor, (ycoor - 1)] + self.grayimg[(xcoor + 1), (ycoor - 1)] -
        #                             self.grayimg[xcoor, (ycoor + 1)] - self.grayimg[(xcoor + 1), (ycoor + 1)]) / 4
        #                     elif r == 0 and c == 2:
        #                         cost = np.abs(
        #                             self.grayimg[xcoor, (ycoor - 1)] - self.grayimg[(xcoor + 1), ycoor]) / np.sqrt(2)
        #                         # cost = np.abs(neighImg[0, 1] - neighImg[1, 2]) / np.sqrt(2)
        #                     elif r == 1 and c == 0:
        #                         cost = np.abs(
        #                             self.grayimg[(xcoor - 1), ycoor] + self.grayimg[(xcoor - 1), (ycoor + 1)] -
        #                             self.grayimg[(xcoor + 1), ycoor] - self.grayimg[(xcoor + 1), (ycoor + 1)]) / 4
        #                         # cost = np.abs(neighImg[0, 0] + neighImg[0, 1] - neighImg[2, 0] - neighImg[2, 1]) / 4
        #                     elif r == 1 and c == 1:
        #                         continue
        #                     elif r == 1 and c == 2:
        #                         cost = np.abs(
        #                             self.grayimg[(xcoor - 1), (ycoor - 1)] + self.grayimg[(xcoor - 1), ycoor] -
        #                             self.grayimg[(xcoor + 1), (ycoor - 1)] - self.grayimg[(xcoor + 1), ycoor]) / 4
        #                         # cost = np.abs(neighImg[0, 1] + neighImg[0, 2] - neighImg[2, 1] - neighImg[2, 2]) / 4
        #                     elif r == 2 and c == 0:
        #                         cost = np.abs(
        #                             self.grayimg[(xcoor-1), ycoor] - self.grayimg[xcoor, (ycoor+1)]) / np.sqrt(2)
        #                         # cost = np.abs(neighImg[1, 0] - neighImg[2, 1]) / np.sqrt(2)
        #                     elif r == 2 and c == 1:
        #                         cost = np.abs(
        #                             self.grayimg[(xcoor - 1), (ycoor - 1)] + self.grayimg[xcoor, (ycoor - 1)] -
        #                             self.grayimg[(xcoor - 1), (ycoor + 1)] - self.grayimg[xcoor, (ycoor + 1)]) / 4
        #                         # cost = np.abs(neighImg[1, 0] + neighImg[2, 0] - neighImg[1, 2] - neighImg[2, 2]) / 4
        #                     elif r == 2 and c == 2:
        #                         cost = np.abs(
        #                             self.grayimg[(xcoor - 1), ycoor] - self.grayimg[xcoor, (ycoor - 1)]) / np.sqrt(2)
        #                         # cost = np.abs(neighImg[1, 2] - neighImg[2, 1]) / np.sqrt(2)
        #                     if maxCost < cost:
        #                         maxCost = cost
        #                     costList[(xcoor, ycoor)] = float(cost)
        #                 except IndexError:
        #                     continue
        #     # costList = {x:(maxCost-y) for x, y in costList.iteritems()}
        #     epsilon = 0.0001
        #     for i, j in costList.iteritems():
        #         if abs(x - i[0]) + abs(y - i[1]) == 1:
        #             costList[i] = maxCost - j + epsilon
        #         else:
        #             costList[i] = (maxCost - j + epsilon) * np.sqrt(2)
        #     return costList

    # def show_contour(self):
    #
    #
    #
    # def pixelNode(self):
    #
    # def costGraph(self):
    #
    #
    # def pathTree(self):
    #
    #
    # def minPath(self):

root = Tkinter.Tk()
root.title('COMP5421 proj1')
app = Scissor(root)
root.mainloop()