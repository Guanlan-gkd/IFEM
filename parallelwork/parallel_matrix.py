from joblib import Parallel, delayed
import numpy as np
import time
import math

im_out = []
num_jobs = 1
img = np.ones((100, 100, 3))
flow = np.zeros((100, 100, 2))
o_grid = np.zeros(flow.shape)

for i in range(0, (flow.shape)[0]):
    o_grid[i,:,0] = i * np.ones((1,(flow.shape)[1]))
    o_grid[i,:,1] = range(0,(flow.shape)[1])
# print(o_grid)

def find(x, i):
    if x >0:
        return i

def GenSpace(flow, o_grid):
    
    change_grid = o_grid + flow
    
    h = (flow.shape)[0]
    w = (flow.shape)[1]
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
    
    spacePt[:,:,0,:] = o_grid[:-1,:-1,:] - Temp1
    spacePt[:,:,1,:] = o_grid[:-1,:-1,:] - Temp2
    spacePt[:,:,2,:] = o_grid[:-1,:-1,:] - Temp3
    spacePt[:,:,3,:] = o_grid[:-1,:-1,:] - Temp4
    
    # print(spacePt)
    
    space = [spaceRect, spacePt] 
    
    return space

def search(space):
    global num_jobs
    
    spaceRect = space[0]
    spacePt = space[1]
    
    A = spaceRect.reshape((-1,4,2))
    B = spacePt.reshape((-1,4,2))
    
    temp = B[:,:,0].copy()
    B[:,:,0] = -B[:,:,1].copy()
    B[:,:,1] = temp.copy()
    
    xP = np.zeros(((A.shape)[0],4))
    
    xP[:,0] = np.multiply(A[:,0,0],B[:,0,0]) + np.multiply(A[:,0,1],B[:,0,1]) 
    xP[:,1] = np.multiply(A[:,1,0],B[:,1,0]) + np.multiply(A[:,1,1],B[:,1,1]) 
    xP[:,2] = np.multiply(A[:,2,0],B[:,2,0]) + np.multiply(A[:,2,1],B[:,2,1]) 
    xP[:,3] = np.multiply(A[:,3,0],B[:,3,0]) + np.multiply(A[:,3,1],B[:,3,1]) 

    L = (xP >= 0) 
    R = (xP <= 0)
    
    L1 = np.multiply(L[:,0],L[:,1],L[:,2],L[:,3])
    R1 = np.multiply(R[:,0],R[:,1],R[:,2],R[:,3])
    
    Result = np.logical_or(L1, R1)
    
    t1 = time.time()
    idx = Parallel(n_jobs=num_jobs)(delayed(find)(Result[i],i) for i in range((Result.shape)[0]))
    print(idx)
    print(time.time() - t1)  
    
    
    
space = GenSpace(flow, o_grid)


