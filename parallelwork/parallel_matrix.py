from joblib import Parallel, delayed
import numpy as np
import time
import math


img = np.ones((100, 100, 3))
flow = np.zeros((100, 100, 2))

im_out = np.zeros(img.shape)
o_grid = np.zeros(flow.shape)

for i in range(0, (flow.shape)[0]):
    o_grid[i,:,0] = i * np.ones((1,(flow.shape)[1]))
    o_grid[i,:,1] = range(0,(flow.shape)[1])
# print(o_grid)

def find(x, i):
    if x >0:
        return i

def GenSpace(change_grid, point, h, w):
    
    c = 2
    
    spaceRect = np.zeros((h-1,w-1,4,c))
    spacePt = np.zeros((h-1,w-1,4,c))
    
    Temp1 = change_grid[:-1,:-1,:]
    Temp2 = change_grid[1:,:-1,:]
    Temp3 = change_grid[1:,1:,:]
    Temp4 = change_grid[:-1,1:,:]
    
    spaceRect[:,:,0,:] = Temp2 - Temp1
    spaceRect[:,:,1,:] = Temp3 - Temp2
    spaceRect[:,:,2,:] = Temp4 - Temp3
    spaceRect[:,:,3,:] = Temp1 - Temp4
    
    # print(spaceRect)
    
    
    spacePt[:,:,0,:] = point - Temp1
    spacePt[:,:,1,:] = point - Temp2
    spacePt[:,:,2,:] = point - Temp3
    spacePt[:,:,3,:] = point - Temp4
    
    # print(spacePt)
    
    space = [spaceRect, spacePt] 
    
    return space

def search(img, change_grid, point, h ,w, o_new):
    global im_out
    n_jobs = 1

    space = GenSpace(change_grid, point, h, w)
    
    spaceRect = space[0]
    spacePt = space[1]
    
    A = np.reshape(spaceRect,(-1,4,2))
    B = np.reshape(spacePt,(-1,4,2))
    
    # temp = B[:,:,0]
    # B[:,:,0] = -B[:,:,1]
    # B[:,:,1] = temp
    
    xP = np.zeros(((A.shape)[0],4))
    
    xP[:,0] = np.multiply(A[:,0,0],B[:,0,1]) - np.multiply(A[:,0,1],B[:,0,0]) 
    xP[:,1] = np.multiply(A[:,1,0],B[:,1,1]) - np.multiply(A[:,1,1],B[:,1,0]) 
    xP[:,2] = np.multiply(A[:,2,0],B[:,2,1]) - np.multiply(A[:,2,1],B[:,2,0]) 
    xP[:,3] = np.multiply(A[:,3,0],B[:,3,1]) - np.multiply(A[:,3,1],B[:,3,0]) 

    L = (xP >= 0) 
    R = (xP <= 0)
    
    L1 = np.multiply(np.multiply(L[:,0],L[:,1]),np.multiply(L[:,2],L[:,3]))
    R1 = np.multiply(np.multiply(R[:,0],R[:,1]),np.multiply(R[:,2],R[:,3]))
    
    Result = np.reshape(np.logical_or(L1, R1), ((L1.shape)[0],1))
    
    switch = 0
    
    for i in range(0, (Result.shape)[0]):
        
        if Result[i][0] > 0 and switch == 0:
            # print("pt", (point),"new pt:" ,(o_new[i,0], o_new[i,1]))
            R_sum = 1 * Result
            
            print(np.sum(R_sum))
            im_out[int(o_new[i,0]), int(o_new[i,1]), :] = img[int(o_new[i,0]), int(o_new[i,1]), :]
            switch = 0

    
    
def fillin(flow, img, o_grid, num_jobs):
    
    h = (flow.shape)[0]
    w = (flow.shape)[1]
    
    change_grid = o_grid + flow
    o_new = np.reshape(o_grid[:h-1,:w-1,:],(-1,2))
    
    
    t1 = time.time()
    Parallel(n_jobs=num_jobs)(delayed(search)(img, change_grid, point, h ,w, o_new) for point in o_new)
    print(time.time() - t1)  

fillin(flow, img, o_grid, num_jobs = 8)