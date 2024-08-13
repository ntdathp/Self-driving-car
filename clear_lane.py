import cv2
import numpy as np


def clear_half(img): #1: clear left;  -1: clear right
   W,H = len(img[0,:]),len(img[:,0])
   yt = 0
   for i in range(60,H):
      S = 0
      yt = 0
      for j in range(W):
         S += img[i,j]
         if S <5:
            yt = j
      S = S/255 
      for j in range(W):
         if (yt-5+S/2) < 50 or yt>200:
            break
         if j<(yt-3+S/2):
            img[i,j] = 0

def clear_t(img,index,thres,alpha): #index = 1: clear left ; = 0: both ; ...
   S = 0 
   it = 0
   W,H = len(img[0,:]),len(img[:,0])
   for i in range(H):
      S = S/W
      if S> thres:
         it = i    
         break
      S = 0
      for j in range(W):
         S += img[i,j]
   scale_y = alpha*it*W/H
   if index == 0:
      for i in range(it-5,H-1):
         for j in range(W):
            k = scale_y+H-10
            if i < (H*k-W*it)/(k-W)+j*(H-it)/(W-k):
               img[i,j] =0
      for i in range(it-5,H-1):
         for j in range(W):
            if i < H+j*(it-H)/(W/2-scale_y):
               img[i,j] =0
   elif index == 1:
      for i in range(it-5,H-1):
         for j in range(W):
            if i < H+j*(it-H)/(W/2-scale_y):
               img[i,j] =0
      
   else: 
      for i in range(it-5,H-1):
         for j in range(W):
            k = scale_y+H-20
            if i < (H*k-W*it)/(k-W)+j*(H-it)/(W-k):
               img[i,j] =0
def clear_top(img,thres):
   S = 0 
   W,H = len(img[0,:]),len(img[:,0])
   for i in range(H):
      S = S/W
      if S > thres:
         break
      S = 0
      for j in range(W):
         S += img[i,j]
         img[i,j] = 0
   if np.sum(img[0:100,1:5])>  np.sum(img[0:100,W-5:W-1]):
      return 1
   else:
      return 0

