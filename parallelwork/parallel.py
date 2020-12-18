from joblib import Parallel, delayed
import numpy as np
import time

i = []

def ReLu(x):
    if x >= 0:
        return x
    else:
        return 0

def isPontinRect(point, Rect):
    # point = np.reshape(point, (1,2))
    # Rect = np.reshape(Rect, (4,2))
    a = (Rect[1][0] - Rect[0][0])*(point[0][1] - Rect[0][1]) - (Rect[1][1] - Rect[0][1])*(point[0][0] - Rect[0][0])
    b = (Rect[2][0] - Rect[1][0])*(point[0][1] - Rect[1][1]) - (Rect[2][1] - Rect[1][1])*(point[0][0] - Rect[1][0])
    c = (Rect[3][0] - Rect[2][0])*(point[0][1] - Rect[2][1]) - (Rect[3][1] - Rect[2][1])*(point[0][0] - Rect[2][0])
    d = (Rect[0][0] - Rect[3][0])*(point[0][1] - Rect[3][1]) - (Rect[0][1] - Rect[3][1])*(point[0][0] - Rect[3][0])
    
    if (a >= 0 and b >= 0 and c > 0 and d > 0) or (a <= 0 and b <=0 and c < 0 and d < 0):
        return 1
    else:
        return 0

def search(point, space):
    ## space is flow in shape (h, w, 2), point is point in shape (1, 2)
    rof = 10
    # h = (space.shape())[0]
    # w = (space.shape())[1]
    point = np.reshape(point,(1,2))

    for i in range(0, int(2*rof)):
        for j in range(0, int(2*rof)):
            ## find the rectangle in origin image
            p1 = [ReLu(point[0][0] + i - rof - 1), ReLU(point[0][0] + j - rof - 1)]
            p2 = [p1[0] + 1, p1[1]]
            p3 = [p1[0] + 1, p1[1] + 1]
            p4 = [p1[0], p1[1] + 1]    
            ## find the transformed rectangle in new image
            p1new = [p1[0] + space[p1[0]][p1[1]][0], p1[1] + space[p1[0]][p1[1]][1]]
            p2new = [p2[0] + space[p2[0]][p2[1]][0], p2[1] + space[p2[0]][p2[1]][1]]
            p3new = [p3[0] + space[p3[0]][p3[1]][0], p3[1] + space[p3[0]][p3[1]][1]]
            p4new = [p4[0] + space[p4[0]][p4[1]][0], p4[1] + space[p4[0]][p4[1]][1]]
            Rect = np.array([p1new, p2new, p3new, p4new])
            point = np.array([point[0][0], point[0][1]])
            if isPontinRect:
                ## return top-left point as the new color refer
                # ence
                return p1

def fillin(point, img, flow):
    global i
    p = search(point, flow)
    i[point[0]][point[1]] = img[p[0]][p[1]]
            

def transform(num_jobs, img, flow):
    global i
    i = np.zeros(img.shape)
    h = (flow.shape())[0]
    w = (flow.shape())[1]

    # h = 2
    # w = 2

    o_grid = np.zeros((h, w, 2))

    for i in range(0, h):
        for j in range(0, w):
            o_grid[i][j][0] = i
            o_grid[i][j][1] = j

    # print(o_grid)
    t1 = time.time()
    Parallel(n_jobs=num_jobs)(delayed(fillin)(point,img,flow) for point in o_grid)
    print(time.time() - t1)
    
