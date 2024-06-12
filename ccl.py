import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

op = os.path.join

def find_frame(a, parent):
    if a == parent[a]:
        return a
    parent[a] = find_frame(parent[a], parent)
    return parent[a]

def ccl(frame):
    n = len(frame)
    m = len(frame[0])
    # dxy = [[-1, 0], [0, -1]] <- 4
    dxy = [[-1, 0], [0, -1], [-1, -1], [-1, 1]]
    parent = [0]
    cnt = 1
    res = frame.copy()
    for i in range(n):
        for j in range(m):
            res[i][j] = 0
            if frame[i][j] == 0:
                continue
            a = []
            for p in dxy:
                ni = i + p[0]
                nj = j + p[1]
                if(0 <= ni and ni < n and 0 <= nj and nj < m and res[ni][nj] != 0):
                    a.append(res[ni][nj])
            if len(a) == 0:
                parent.append(cnt)
                res[i][j] = cnt
                cnt += 1
            else:
                res[i][j] = min(a)
                for p in a:
                    pa = find_frame(p, parent)
                    pb = find_frame(res[i][j], parent)
                    if(pa > pb) :
                        pa, pb = pb, pa
                    parent[pb] = pa
    component_cnt = [0 for i in range(len(parent))]
    cnt = 1
    for i in range(1, len(parent)):
        if component_cnt[find_frame(i, parent)] == 0:
            component_cnt[find_frame(i, parent)] = cnt
            cnt += 1
        component_cnt[i] = component_cnt[find_frame(i, parent)]
    for i in range(n):
        for j in range(m):
            if frame[i][j] != 0:
                res[i][j] = component_cnt[res[i][j]] 
    return res 

def show_labeled_image(img):
    # different random colors for each label
    cmap = plt.cm.get_cmap('tab20', img.max())
    plt.imshow(img, cmap=cmap)
    plt.show()
    
def labeled_image(img):
    # 각 라벨에 대해 다른 랜덤 색상 생성
    cmap = plt.cm.get_cmap('tab20', img.max())
    
    # 이미지를 색상 맵에 맞게 변환
    labeled_image = cmap(img / img.max())
    
    # Alpha 채널 제거 (옵션)
    labeled_image = labeled_image[:, :, :3]
    
    return (labeled_image * 255).astype(np.uint8)