# this code is from Lyu Chaofan 1706987, without any changes

import sys, time
import numpy as np
import cv2

class dataSet:
    def __init__(self):
        self.data  = []
        self.label = []
        self.dic   = {}
        
    def upData(self, data, label, dic):
        self.data.append(data)
        self.label.append(label)
        self.dic = dic

    def sHape(self):
        datashape = self.data[0].shape
        row, col, lay = datashape
        result = np.prod(np.array(datashape)) 
        data = self.data[0].reshape(1, result)
        process_bar = ShowProcess(len(self.data)-1)
        for i in range(1, len(self.data)):
            data = np.r_[data, self.data[i].reshape(1, result)]
            # process_bar.show_process()
        process_bar.close()
        self.data = data
        self.label = np.array(self.label)
        
class ShowProcess():

    i = 0
    max_steps = 0
    max_arrow = 50

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0

    def close(self, words='done'):
        # print ('')
        # print (words)
        self.i = 0

if __name__ == '__main__':
    image_data = cv2.imread('./Original_images/amazon/images/ruler/frame_0001.jpg')
    # print (image_data)
    labels = {'printer': 21, 
              'projector': 22, 
              'file_cabinet': 9, 
              'ruler': 25, 
              'trash_can': 30, 
              'phone': 20, 
              'bottle': 4, 
              'laptop_computer': 12, 
              'bookcase': 3, 
              'letter_tray': 13, 
              'back_pack': 0, 
              'paper_notebook': 18, 
              'calculator': 5, 
              'desk_chair': 7, 
              'mug': 17, 
              'pen': 19, 
              'monitor': 15, 
              'mouse': 16, 
              'desktop_computer': 6, 
              'ring_binder': 24, 
              'stapler': 28, 
              'bike_helmet': 2, 
              'tape_dispenser': 29, 
              'desk_lamp': 8, 
              'keyboard': 11, 
              'bike': 1, 
              'punchers': 23, 
              'mobile_phone': 14, 
              'scissors': 26, 
              'headphones': 10, 
              'speaker': 27}
    
     
    test = dataSet()
    test.upData(image_data, labels['ruler'], labels)
    test.upData(image_data, labels['ruler'], labels)
    test.upData(image_data, labels['ruler'], labels)
    test.upData(image_data, labels['ruler'], labels)
    test.sHape()
    # print(test.data.shape)
    # print(test.label.shape)
