
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import sys
import time


def center_point(p1, p2):
    # 输入p1,p2,返回它的中心点
    cp = []
    for i in range(3):
        cp.append((p1[i] + p2[i]) / 2)

    return cp


def sum_point(p1, p2):
    # 求p1,p2的和
    sp = []
    for i in range(3):
        sp.append(p1[i] + p2[i])

    return sp


def div_point(p, d):
    #用d除以坐标p
    sp = []
    for i in range(3):
        sp.append(p[i] / d)
    return sp


def mul_point(p, m):
    #用m乘p
    sp = []
    for i in range(3):
        sp.append(p[i] * m)

    return sp


def get_face_points(input_points, input_faces):
    # 计算面点,对每一个面,先求其顶点的和,再除以其顶点个数(len(curr_face))
    NUM_DIMENSIONS = 3

    face_points = []

    for curr_face in input_faces:
        face_point = [0.0, 0.0, 0.0]
        for curr_point_index in curr_face:
            curr_point = input_points[curr_point_index]
            for i in range(NUM_DIMENSIONS):
                face_point[i] += curr_point[i]
        # divide by number of points for average
        num_points = len(curr_face)
        for i in range(NUM_DIMENSIONS):
            face_point[i] /= num_points
        face_points.append(face_point)

    return face_points


def get_edges_faces(input_points, input_faces):
    # 计算边点的辅助函数,除了要计算边的中间点,还要判断这条边是两个面的交线还是边线,并将其相邻的1或两个面的编号记录
    edges = []

    # 先遍历所有面,找到所有的边(如果两顶点相同,但面不同,我们算不同的边,将其存入)

    for facenum in range(len(input_faces)):
        face = input_faces[facenum]
        num_points = len(face)
        # loop over index into face
        for pointindex in range(num_points):
            # if not last point then edge is curr point and next point
            if pointindex < num_points - 1:
                pointnum_1 = face[pointindex]
                pointnum_2 = face[pointindex + 1]
            else:
                # for last point edge is curr point and first point
                pointnum_1 = face[pointindex]
                pointnum_2 = face[0]
            # order points in edge by lowest point number
            if pointnum_1 > pointnum_2:
                temp = pointnum_1
                pointnum_1 = pointnum_2
                pointnum_2 = temp
            edges.append([pointnum_1, pointnum_2, facenum])

    # 对边进行排序,优先级为: pointnum_1, pointnum_2, facenum
    # 这样,如果顶点相同的两边会被放到一起,如果facenum不同,说明其是由两面相交形成的
    edges = sorted(edges)

    # 记录边,两种形式其一为交线,其二为边缘边
    # [pointnum_1, pointnum_2, facenum_1, facenum_2] or
    # [pointnum_1, pointnum_2, facenum_1, None]

    num_edges = len(edges)
    eindex = 0
    merged_edges = []

    while eindex < num_edges:
        e1 = edges[eindex]
        # check if not last edge
        if eindex < num_edges - 1:
            e2 = edges[eindex + 1]
            if e1[0] == e2[0] and e1[1] == e2[1]:
                merged_edges.append([e1[0], e1[1], e1[2], e2[2]])
                eindex += 2
            else:
                merged_edges.append([e1[0], e1[1], e1[2], None])
                eindex += 1
        else:
            merged_edges.append([e1[0], e1[1], e1[2], None])
            eindex += 1

    # add edge centers

    edges_centers = []

    for me in merged_edges:
        p1 = input_points[me[0]]
        p2 = input_points[me[1]]
        cp = center_point(p1, p2)
        edges_centers.append(me + [cp])# [pointnum_1, pointnum_2, facenum_1, facenum_2,center_point],facenum_2可能为NULL

    return edges_centers


def get_edge_points(input_points, edges_faces, face_points):
    # 计算边点时,要用到:边的中心点,上个函数已求,相邻的两个面的面点的中心点
    # 注意:对于边缘的边,我们用其中心点替代边点
    edge_points = []

    for edge in edges_faces:
        # get center of edge
        cp = edge[4]
        # get center of two facepoints
        fp1 = face_points[edge[2]]
        # if not two faces just use one facepoint
        # should not happen for solid like a cube
        if edge[3] == None:
            fp2 = fp1 = cp
        else:
            fp2 = face_points[edge[3]]
        cfp = center_point(fp1, fp2)
        # get average between center of edge and
        # center of facepoints
        edge_point = center_point(cp, cfp)
        edge_points.append(edge_point)

    return edge_points


def get_avg_face_points(input_points, input_faces, face_points):
    # 计算平均面点,我们要确认一个点属于多少个面,并一次性将这些面的面点求个和,后面直接用面点和除面数即可

    # initialize list with [[0.0, 0.0, 0.0], 0]

    num_points = len(input_points)

    temp_points = []

    for pointnum in range(num_points):
        temp_points.append([[0.0, 0.0, 0.0], 0])

    # loop through faces updating temp_points

    for facenum in range(len(input_faces)):
        fp = face_points[facenum]
        for pointnum in input_faces[facenum]:
            tp = temp_points[pointnum][0]
            temp_points[pointnum][0] = sum_point(tp, fp)
            temp_points[pointnum][1] += 1

#以上统计每个点属于多少个面，并将每个面点求和，在接下来求平均
    # divide to create avg_face_points
    """
    print(len(temp_points))
    for i in range(len(temp_points)):
        print(temp_points[i])
    """
    avg_face_points = []

    for tp in temp_points:
        afp = div_point(tp[0], tp[1])
        avg_face_points.append(afp)

    return avg_face_points


def get_avg_mid_edges(input_points, edges_faces):
    #计算平均边中点,同上一个函数,不过这里要确认一个点有多少条边

    # initialize list with [[0.0, 0.0, 0.0], 0]

    num_points = len(input_points)

    temp_points = []

    for pointnum in range(num_points):
        temp_points.append([[0.0, 0.0, 0.0], 0])

    # go through edges_faces using center updating each point

    for edge in edges_faces:
        cp = edge[4]
        for pointnum in [edge[0], edge[1]]:
            tp = temp_points[pointnum][0]
            temp_points[pointnum][0] = sum_point(tp, cp)
            temp_points[pointnum][1] += 1

    # divide out number of points to get average

    avg_mid_edges = []

    for tp in temp_points:
        ame = div_point(tp[0], tp[1])
        avg_mid_edges.append(ame)

    return avg_mid_edges


def get_points_faces(input_points, input_faces):
    # 单独设计一个函数,统计一个原始点有多少个相邻面

    num_points = len(input_points)

    points_faces = []

    for pointnum in range(num_points):
        points_faces.append(0)

    # loop through faces updating points_faces

    for facenum in range(len(input_faces)):
        for pointnum in input_faces[facenum]:
            points_faces[pointnum] += 1

    return points_faces
#统计每个点属于多少个面

def get_new_points(input_points, points_faces, avg_face_points, avg_mid_edges):
    # 利用上述函数,我们已经知道了一个点有多少个相邻面,以及其平均面点,平均边中电
    # 注意,当面数小于二时,说明其为边缘点,我们直接用原坐标替代更新后的坐标
    new_points = []

    for pointnum in range(len(input_points)):

        n = points_faces[pointnum]
        if n>=3:
            m1 = (n - 3.0) / n
            m2 = 1.0 / n
            m3 = 2.0 / n
            old_coords = input_points[pointnum]
            p1 = mul_point(old_coords, m1)
            afp = avg_face_points[pointnum]
            p2 = mul_point(afp, m2)
            ame = avg_mid_edges[pointnum]
            p3 = mul_point(ame, m3)
            p4 = sum_point(p1, p2)
            new_coords = sum_point(p4, p3)
        else:
            new_coords = input_points[pointnum]

        new_points.append(new_coords)

    return new_points


def switch_nums(point_nums):
    # 生成新面时,要用到边点,我们在求边中点的时候,找到了每条边的两个顶点,我们对其废物利用,因为在哪里边时由小到大排好序的,所以在这里也排一下.
    if point_nums[0] < point_nums[1]:
        return point_nums
    else:
        return (point_nums[1], point_nums[0])


def cmc_subdiv(input_points, input_faces):
    # 正式进行细分的函数
    # 1. for each face, a face point is created which is the average of all the points of the face.
    # each entry in the returned list is a point (x, y, z).

    face_points = get_face_points(input_points, input_faces)
    print(face_points)
    print("---------------face_points")

    # get list of edges with 1 or 2 adjacent faces
    # [pointnum_1, pointnum_2, facenum_1, facenum_2, center] or
    # [pointnum_1, pointnum_2, facenum_1, None, center]

    edges_faces = get_edges_faces(input_points, input_faces)
    print(edges_faces)
    print("---------------edges_faces")

    # get edge points, a list of points

    edge_points = get_edge_points(input_points, edges_faces, face_points)
    print(edge_points)
    print("---------------edge_points")

    # the average of the face points of the faces the point belongs to (avg_face_points)

    avg_face_points = get_avg_face_points(input_points, input_faces, face_points)
    print(avg_face_points)
    print("---------------avg_face_points")

    # the average of the centers of edges the point belongs to (avg_mid_edges)

    avg_mid_edges = get_avg_mid_edges(input_points, edges_faces)
    print(avg_mid_edges)
    print("---------------avg_mid_edges")

    # how many faces a point belongs to

    points_faces = get_points_faces(input_points, input_faces)

    """

    m1 = (n - 3) / n
    m2 = 1 / n
    m3 = 2 / n
    new_coords = (m1 * old_coords)
               + (m2 * avg_face_points)
               + (m3 * avg_mid_edges)

    """
    # 在这里得到更新后的点
    new_points = get_new_points(input_points, points_faces, avg_face_points, avg_mid_edges)
    print(new_points)
    print("---------------new_points1")


    # 把面点也加进新点

    face_point_nums = []

    # point num after next append to new_points
    next_pointnum = len(new_points)

    for face_point in face_points:
        new_points.append(face_point)
        face_point_nums.append(next_pointnum)# 记录旧面的顶点在新点集合中的编号
        next_pointnum += 1
    print(new_points)
    print("---------------new_points2")
    # 把边点也加进新点

    edge_point_nums = dict()

    for edgenum in range(len(edges_faces)):
        pointnum_1 = edges_faces[edgenum][0]
        pointnum_2 = edges_faces[edgenum][1]
        edge_point = edge_points[edgenum]
        new_points.append(edge_point)
        edge_point_nums[switch_nums((pointnum_1,pointnum_2))] = next_pointnum# 记录边点在新集合中的编号(边点由其两端的端点确定)
        next_pointnum += 1
    print(new_points)
    print("---------------new_points3")
    # new_points now has the points to output. Need new
    # faces

    # 从原面更新出新面

    new_faces = []

    for oldfacenum in range(len(input_faces)):
        oldface = input_faces[oldfacenum]
        # 4 point face
        if len(oldface) == 4:
            a = oldface[0]
            b = oldface[1]
            c = oldface[2]
            d = oldface[3]
            face_point_abcd = face_point_nums[oldfacenum]
            edge_point_ab = edge_point_nums[switch_nums((a, b))]
            edge_point_da = edge_point_nums[switch_nums((d, a))]
            edge_point_bc = edge_point_nums[switch_nums((b, c))]
            edge_point_cd = edge_point_nums[switch_nums((c, d))]
            new_faces.append((a, edge_point_ab, face_point_abcd, edge_point_da))
            new_faces.append((b, edge_point_bc, face_point_abcd, edge_point_ab))
            new_faces.append((c, edge_point_cd, face_point_abcd, edge_point_bc))
            new_faces.append((d, edge_point_da, face_point_abcd, edge_point_cd))
        # 3 point face
        if len(oldface) == 3:
            a = oldface[0]
            b = oldface[1]
            c = oldface[2]
            face_point_abcd = face_point_nums[oldfacenum]
            edge_point_ab = edge_point_nums[switch_nums((a, b))]
            edge_point_ca = edge_point_nums[switch_nums((c, a))]
            edge_point_bc = edge_point_nums[switch_nums((b, c))]
            new_faces.append((a, edge_point_ab, face_point_abcd, edge_point_ca))
            new_faces.append((b, edge_point_ab, face_point_abcd, edge_point_bc))
            new_faces.append((c, edge_point_ca, face_point_abcd, edge_point_bc))

    return new_points, new_faces


def graph_output(output_points, output_faces, fig):

    ax = fig.add_subplot(111, projection='3d')


    """

    Plot each face

    """

    for facenum in range(len(output_faces)):
        curr_face = output_faces[facenum]
        xcurr = []
        ycurr = []
        zcurr = []
        for pointnum in range(len(curr_face)):
            xcurr.append(output_points[curr_face[pointnum]][0])
            ycurr.append(output_points[curr_face[pointnum]][1])
            zcurr.append(output_points[curr_face[pointnum]][2])
        xcurr.append(output_points[curr_face[0]][0])
        ycurr.append(output_points[curr_face[0]][1])
        zcurr.append(output_points[curr_face[0]][2])

        ax.plot(xcurr, ycurr,zcurr,color='g')





# cube

input_points = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [3.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [2.0, 1.0, 0.0],#6
    [3.0, 1.0, 0.0],
    [0.0, 2.0, 0.0],
    [1.0, 2.0, 0.0],
    [2.0, 2.0, 0.0],
    [3.0, 2.0, 0.0],
    [0.0, 3.0, 0.0],
    [1.0, 3.0, 0.0],
    [2.0, 3.0, 0.0],
    [3.0, 3.0, 0.0]
]

input_faces = [
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [4, 5, 9, 8],
    [5, 6, 10],
    [5, 9, 10],
    #[5, 6, 10, 9],
    [6, 7, 11, 10],
    [8, 9, 13, 12],
    [9, 10, 14, 13],
    [10, 11, 15, 14]

]
"""
input_points=[
    [0.0,0.0,0.0],
    [1.0,0.0,0.0],
    [1.0,1.0,0.0],
    [0.0,1.0,0.0],
]
input_faces =[
    [0,1,2],
    [0,2,3]
]
"""
iterations = 4

plt.ion()
output_points, output_faces = input_points, input_faces
print(output_points)
print("---------------output_points")
print(output_faces)
print("---------------output_faces")
fig = plt.figure(1)
plt.clf()
graph_output(output_points, output_faces, fig)
plt.pause(1)
print("---------------------------------------")
for i in range(iterations):
    output_points, output_faces = cmc_subdiv(output_points, output_faces)
    print(output_points)
    print(output_faces)
    fig = plt.figure(1)
    plt.clf()
    graph_output(output_points, output_faces, fig)
    plt.pause(1)
    print("---------------------------------------")