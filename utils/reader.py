import numpy as np
import torch

def readOBJ(filepath, returnType = 'numpy', dtype=torch.float, device = torch.device('cpu')):
    ## read mesh from .obj file
    ##
    ## Inputs:
    ## filepath: file to obj file
    ##
    ## Outputs:
    ## V: n-by-3 numpy ndarray of vertex positions
    ## F: m-by-3 numpy ndarray of face indices
    V = []
    # F = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    while True:
        for line in lines:
            if line == "":
                break
            elif line.strip().startswith("v "):
                vertices = line.replace("\n", "").split(" ")[1:4]
                vertices = np.delete(vertices,np.argwhere(vertices == np.array([''])).flatten())
                # print ([float(i) for i in vertices])
                V.append([float(i) for i in vertices])
            # elif line.strip().startswith("vn"):
            #     continue
            # elif line.strip().startswith("vt"):
            #     continue
            # elif line.strip().startswith("f"):
            #     t_index_list = []
            #     for t in line.replace("\n", "").split(" ")[1:]:
            #         t_index = t.split("/")[0]
            #         try: 
            #             t_index_list.append(int(t_index) - 1)
            #         except ValueError:
            #             continue
            #     F.append(t_index_list)
            else:
                continue
        break

    if returnType == 'torch':
        V = torch.tensor(V, dtype=dtype, device=device)
        # F = torch.tensor(F, dtype=torch.long, device=device)
    else:
        V = np.array(V)
        # F = np.array(F)
    
    return V, None

def readOBJsize(filepath):

    size = 0

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.strip().startswith("v"):
            size += 1

    return size

def readPTS(filepath):
    ## read pointcloud from .pts file
    ##
    ## Inputs:
    ## filepath: file to .pts file
    ##
    ## Outputs:
    ## V: n-by-3 numpy ndarray of vertex positions
    V = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    while True:
        for line in lines:
            if line == "":
                break
            else:
                vertices = line.replace("\n", "").split(" ")
                # vertices = np.delete(vertices,np.argwhere(vertices == np.array([''])).flatten())
                V.append([float(i) for i in vertices])
        break

    V = np.array(V)

    return V