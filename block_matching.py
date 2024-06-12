import numpy as np

def blockMatching(preframe, curframe):
    n = len(preframe)
    m = len(preframe[0])
    # cnt = 0
    
    #label 개수 구하기
    # for i in range(n):
    #     for j in range(m):
    #         cnt = max(cnt, preframe[i][j])
    cnt = 1

    # midpoint = np.array([(-1,-1) for i in range(cnt+1)])
    midpoint = [-1, -1]
    # pointcnt = np.array([0 for i in range(cnt+1)])
    pointcnt = 0
    
    #각 component의 중심 구하기
    for i in range(n):
        for j in range(m):
            curnum = preframe[i][j]
            if curnum == 0:
                continue
            pointcnt += 1
            midpoint[0] += i
            midpoint[1] += j
    for i in range(1,cnt+1):
        if(pointcnt == 0) :
            continue
        midpoint[0] = (int)(midpoint[0] // pointcnt)
        midpoint[1] = (int)(midpoint[1] // pointcnt)
        
    #각 component의 중심에서 block matching (three step)
    dy = [0,0,0,-1,-1,-1,1,1,1]
    dx = [0,1,-1,0,1,-1,0,1,-1]

    if pointcnt == 0:
        return [0, 0]

    midy = midpoint[0]
    midx = midpoint[1]
    oriy = midy
    orix = midx
    step = 16
    
    while(step >= 2):
        maxsum = -1
        nextdy = 0
        nextdx = 0
        for j in range(9):
            ny = midy + dy[j]
            nx = midx + dx[j]
            sum = 0
            for y in range(-step//2, step//2+1):
                for x in range(-step//2, step//2+1):
                    if(preframe[oriy+y][oriy+x] == 0 and curframe[ny+y][nx+x] == 0):
                        sum += 1
                    elif preframe[oriy+y][oriy+x] > 0 and curframe[ny+y][nx+x] > 0:
                        sum += 2
            if(maxsum < sum):
                maxsum = sum
                nextdy = dy[j]
                nextdx = dx[j]
        step = step // 2
        midy = midy + nextdy * step
        midx = midx + nextdx * step
    res = (midy - midpoint[0], midx - midpoint[1])
    
    return res

def preprocess_image(image, target_label):
    """
    Convert all components in the image to 0 except for the target_label.
    
    Args:
    image (np.array): The labeled image array.
    target_label (int): The label to keep.
    
    Returns:
    np.array: The processed image with only the target_label.
    """
    # Create a copy of the image to avoid modifying the original
    processed_image = np.copy(image)
    
    # Set all non-target_label values to 0
    processed_image[processed_image != target_label] = 0
    
    return processed_image