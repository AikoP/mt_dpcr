import sys
import os
sys.path.insert(1, os.getcwd())  # to load from any submodule in the repo

import numpy as np
# import torch
from utils import readOBJ
from utils import writeOBJ

def sampleOBJ(in_path, out_path):

    # device = torch.device("cpu")

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print("Using %d CUDA devices (%s)" % (torch.cuda.device_count(), torch.cuda.get_device_name(0)))
    # else:
    #     print('Using CPU')

    V, F = readOBJ.readOBJ(in_path)

    marks = np.zeros((F.shape[0],F.shape[0]))

    edge_faces = []
    edge_vertices = []
    edge_lengths = []
    for i in range(F.shape[0]):
        for (a,b) in [(0,1),(1,2),(0,2)]:
            v1, v2 = F[i,a], F[i,b]
            length = np.sum(np.square(V[v1]-V[v2]))
            for j in range(F.shape[0]):
                if i != j and v1 in F[j] and v2 in F[j]:
                    if marks[i,j] != 1:
                        edge_vertices.append([v1, v2])
                        edge_lengths.append(length)
                        edge_faces.append([i,j])
                        marks[i,j] = 1
                        marks[j,i] = 1
                    break

    order = np.flip(np.array(edge_lengths).argsort())
    edge_faces = np.array(edge_faces)[order].tolist()
    edge_vertices = np.array(edge_vertices)[order].tolist()
    edge_lengths = np.array(edge_lengths)[order].tolist()

    print (edge_vertices)

    # V = torch.from_numpy(V).to(device)
    # F = torch.from_numpy(F).to(device).long()

    for k in range(1000):

        length = edge_lengths.pop(0)

        vertices = edge_vertices.pop(0)
        v1 = vertices[0]
        v2 = vertices[1]

        # add new vertex
        V = np.append(V, 0.5 * (V[v1] + V[v2]).reshape((1,-1)), axis=0)

        faces = edge_faces.pop(0)

        new_edges_length = [length / 2, length / 2]
        new_edges_faces = [faces, [F.shape[0], F.shape[0] + 1]]
        new_edge_vertices = [[v1, V.shape[0] - 1], [v2, V.shape[0] - 1]]
        
        for f in faces:
            op = 0
            for j in range(3):
                if F[f][j] != v1 and F[f][j] != v2:
                    op = F[f][j]

            new_face_1 = np.array([op, v1, V.shape[0] - 1])
            new_face_2 = np.array([op, v2, V.shape[0] - 1])

            F[f] = new_face_1
            F = np.append(F, new_face_2.reshape((1,-1)), axis = 0)

            new_edges_length.append(np.sum(np.square(V[op]-V[-1])))
            new_edges_faces.append([f,F.shape[0] - 1])
            new_edge_vertices.append([op, V.shape[0] - 1])

            # remap the edge to the new face
            for i in range(len(edge_lengths)):
                if v2 in edge_vertices[i] and op in edge_vertices[i]:
                    for j in range(len(edge_faces[i])):
                        if edge_faces[i][j] == f:
                            edge_faces[i][j] = F.shape[0] - 1
                            break
                    break

        for i in range(len(new_edges_length)):
            if new_edges_length[i] < edge_lengths[-1]:
                edge_lengths.append(new_edges_length[i])
                edge_faces.append(new_edges_faces[i])
                edge_vertices.append(new_edge_vertices[i])
            else:
                insert_pos = 0
                for j in range(len(edge_lengths)):
                    insert_pos = j
                    if edge_lengths[j] < new_edges_length[i]:
                        break
                edge_lengths.insert(insert_pos, new_edges_length[i])
                edge_faces.insert(insert_pos, new_edges_faces[i])
                edge_vertices.insert(insert_pos, new_edge_vertices[i])

        # print (edge_lengths)

        # find the longest edge
        # max_length = -1
        # max_v1 = 0
        # max_v2 = 0
        # for i in range(F.shape[0]):
        #     for (a,b) in [(0,1),(1,2),(0,2)]:
        #         v1, v2 = F[i,a], F[i,b]
        #         length = np.sum(np.square(V[v1]-V[v2]))
        #         if max_length < 0 or length > max_length:
        #             max_v1, max_v2 = v1, v2
        #             max_length = length

        # if (max_length < 0.1):
        #     print ("threshold!")
        #     break

        # # max_v = torch.sum((V[F] - V[F[:,[1,2,0]]]) ** 2, dim = 2).argmax().item()

        # # if max_v % 3 == 0:
        # #     max_v1, max_v2 = F[max_v // 3, 0], F[max_v // 3, 1]
        # # elif max_v % 3 == 1:
        # #     max_v1, max_v2 = F[max_v // 3, 1], F[max_v // 3, 2]
        # # else:
        # #     max_v1, max_v2 = F[max_v // 3, 0], F[max_v // 3, 2]

        # # print(max_v)

        # # collect all faces adjacent to the longest edge
        # splitFaces = []
        # for i in range(F.shape[0]):
        #     if max_v1 in F[i] and max_v2 in F[i]:
        #         splitFaces.append(i)

        # # add new vertex
        # V = np.append(V, 0.5 * (V[max_v1] + V[max_v2]).reshape((1,-1)), axis=0)

        # for i in splitFaces:
        #     opposite_point = 0
        #     for j in range(3):
        #         if F[i,j] != max_v1 and F[i,j] != max_v2:
        #             opposite_point = F[i,j]
            
        #     new_face_1 = np.array([opposite_point, max_v1, V.shape[0] - 1])
        #     new_face_2 = np.array([opposite_point, max_v2, V.shape[0] - 1])

        #     F[i] = new_face_1
        #     F = np.append(F, new_face_2.reshape((1,-1)), axis = 0)

    writeOBJ.writeOBJ(out_path,V,F, writeFaces=False)


if __name__ == "__main__":
    sampleOBJ('D:\\Github\\Repos\\dpcr\\data\\simple_shapes\\base\\cube.obj', 'D:\\Github\\Repos\\dpcr\\data\\simple_shapes\\base\\cube_sampled.obj')