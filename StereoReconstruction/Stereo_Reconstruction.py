
import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import math


def intersectNumpy(a, b):
    common = np.array([value for value in a.tolist() if value in b.tolist()])
    unique = np.unique(common, axis = 0)
    return unique


def isMemberLogical(a, b):
    bind = {}
    for i, elt in enumerate(b): # create a dictionary of each row of b
        if tuple(elt) not in bind:
            bind[tuple(elt)] = True
    return [bind.get(tuple(itm), False) for itm in a] # for each value of a, check if tuple is in b and return True if its there, false if not



def find_match(img1, img2):

    sift = cv2.xfeatures2d.SIFT_create() #nOctaveLayers = 3, contrastThreshold = 0.02)
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)


    #siftImg1=cv2.drawKeypoints(img1,kp1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #siftImg2=cv2.drawKeypoints(img2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # fit a knn classifier to the first image
    nbrs1 = NearestNeighbors(n_neighbors = 2).fit(des1)
    # for each point in the second image, which is closest in the first
    dist2to1, ind2to1 = nbrs1.kneighbors(des2)

    # fit a knn classifier to the second image
    nbrs2 = NearestNeighbors(n_neighbors = 2).fit(des2)
    # for each point in the first image, which is closest in the second
    dist1to2, ind1to2 = nbrs2.kneighbors(des1)

    # build a vector which contains the array index followed by the top match index from 1 to 2
    match1to2 = np.hstack(( np.arange(0, ind1to2.shape[0], 1 ).reshape(ind1to2.shape[0],1)  ,   ind1to2[:,0].reshape(ind1to2.shape[0],1)                        ))
    # build a vector which contains the top match index from 2 to 1 followed by the array index
    match2to1 = np.hstack(( ind2to1[:,0].reshape(ind2to1.shape[0],1)                        ,   np.arange(0, ind2to1.shape[0], 1 ).reshape(ind2to1.shape[0],1)  ))

    intersection = intersectNumpy(match1to2, match2to1) # get common rows

    logicalBidirectional = isMemberLogical( match1to2, intersection ) # bidirectional test (False on bad rows)
    
    # ratio test
    logicalRatio = dist1to2[:,0]/dist1to2[:,1] < .7 # get good ratios
    
    logratcount = 0
    for i in logicalRatio:
        if i:
            logratcount += 1

    logbidircount = 0
    for i in logicalBidirectional:
        if i:
            logbidircount += 1

    # what to include (only points in the bidrectional test that pass the ratio test)
    logicalMembers =    logicalRatio & logicalBidirectional # [True for i in logicalRatio]# logicalRatio & logicalBidirectional &

    # logical trimming
    trueMatches1to2 = match1to2[logicalMembers,:] #ind1to2[logicalMembers,:]

    # get the indices for the matches in the first image, and use those to find the keypoints
    kp1Index = trueMatches1to2[:,0].tolist()
    x1 = np.array([np.array(kp1[idx].pt) for idx in kp1Index])
    
    # get the indices for the matches in the second image, and use those to find the keypoints
    kp2Index = trueMatches1to2[:,1].tolist()
    x2 = np.array([np.array(kp2[idx].pt) for idx in kp2Index])

    return x1, x2


def getF8pt(u,v):
    # should check if u and v are the same size
    A = np.zeros((u.shape[0],9))
    for ii in range( u.shape[0] ):
        A[ii,:] = np.array([u[ii,0]*v[ii,0], u[ii,1]*v[ii,0], v[ii,0], u[ii,0]*v[ii,1], u[ii,1]*v[ii,1], v[ii,1], u[ii,0], u[ii,1], 1])



    u, s, vh = np.linalg.svd(A, full_matrices=True)
    nullSpace = vh[-1,:]
    F = np.array([ nullSpace[:3], nullSpace[3:6], nullSpace[6:] ])

    # SVD cleanup
    u, s, vh = np.linalg.svd(F, full_matrices=True)
    s[2] = 0
    F = np.dot(u, np.dot(np.diag(s),vh))

    return F


def chooseNPoints(x1, x2, n):
       
    choiceSize = n
    p1 = np.zeros((choiceSize,2));
    p2 = np.zeros((choiceSize,2));
    seen = [-1 for i in range(choiceSize)];
    numFeatures = x2.shape[0];
    
    count = 0;
    while count < choiceSize:
        randid = np.random.randint(0,numFeatures);
        # if ~isempty(find(seen==randid)) 
        if any([seen[i] == randid for i in range(len(seen))]): # if the value has been seen and accepted
            continue;
        # if x2[randid,0] != -1 # will always be valid, so commenting this entire line
        seen[count] = randid;
        p1[count,:] = x1[randid,:];
        p2[count,:] = x2[randid,:];
        count = count + 1;

    return p1, p2


def compute_F(pts1, pts2):
    M = 2000 # ransac iterations
    n = 0; # max inliers
    F = np.zeros( ( 3, 3 ) )
    for ii in range(M):
        u_r, v_r = chooseNPoints( pts1, pts2, 8 )
        F_r = getF8pt( u_r, v_r )
        # compute num inliers
        # make sure you're using correct F -- Fu vs F'v
        n_r = 0
        D = 3 # threshold
        for jj in range( pts2.shape[0] ):
            #if Mb[jj,1] == -1
            #    continue;
            #end
            v = (pts2[jj,0],pts2[jj,1],1)
            lv = np.dot(np.transpose(F_r), v) #line from feature v appearing in feature u
            u = pts1[jj]
            num = abs(lv[0]*u[0] + lv[1]*u[1] + lv[2])
            denom = math.sqrt(lv[0]*lv[0] + lv[1]*lv[1])
            d = num/denom
            if d < D:
                n_r = n_r + 1;

        if n_r > n:
            n = n_r
            F = F_r

    return F

def skewPix(x):
    return np.array([[0, -1, x[1]],
                     [1, 0, -x[0]],
                     [-x[1], x[0], 0]])

def triangulation(P1, P2, pts1, pts2):
    pts3D = np.zeros((pts1.shape[0], 4))

    A = np.zeros((6,4))

    for i in range(pts1.shape[0]):
        A[:3,:] = np.dot(skewPix(pts1[i]),P1)
        A[3:,:] = np.dot(skewPix(pts2[i]),P2)
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        nullSpace = vh[-1,:]

        pts3D[i] = nullSpace / nullSpace[3]

    return pts3D

def disambiguate_pose(Rs, Cs, pts3Ds):
    bestValidIdx = -1
    bestValid = 0
    
    RL = np.identity(3)
    CL = np.zeros(3)

    for ii in range( len(Rs) ):
        
        nValid = 0
        RR = Rs[ii]
        CR = np.squeeze(Cs[ii])

        for X in pts3Ds[ii]:
            cheiralityL = np.dot( RL[2,:], X[:3] - CL )
            cheiralityR = np.dot( RR[2,:], X[:3] - CR )
            if cheiralityL > 0 and cheiralityR > 0:
                nValid += 1

        if nValid > bestValid:
            bestValid = nValid
            bestValidIdx = ii

    return Rs[bestValidIdx], Cs[bestValidIdx], pts3Ds[bestValidIdx]


def compute_rectification(K, R, C):
    rx = np.squeeze( C )
    rx /= np.linalg.norm( rx )
    rz_hat = R[2,:]
    rz = rz_hat - ( np.dot( rz_hat, rx ) * rx )
    rz /= np.linalg.norm( rz )
    ry = np.cross( rz, rx )

    R_rect = np.array([rx,ry,rz])

    H1 = K @ R_rect @ np.linalg.inv( K )
    H2 = K @ R_rect @ np.transpose( R ) @ np.linalg.inv( K )

    return H1, H2


def dense_match(img1, img2):
    size = 3
    #cv2.imshow('Left',img1)
    #cv2.imshow('Right',img2)
    #cv2.waitKey()

    disparity = np.zeros(img1.shape, dtype = int)

    #grid = np.indices( ( img1.shape[0], img1.shape[1] ) )

    #keyPoints = [ [ cv2.KeyPoint() ] * img1.shape[1] ] * img1.shape[0]
    #keyPoints = np.empty( img1.shape, dtype = type(cv2.KeyPoint) )

    print('Building keypoints...')
    #v = np.vectorize( lambda a,b: cv2.KeyPoint(a,b,size) )
    #keyPoints = v( grid[1], grid[0] ).flatten().tolist()

    keyPoints = [cv2.KeyPoint()] * img1.shape[0] * img2.shape[1]
    length = len(keyPoints)
    for y in range(img1.shape[0]):
        for x in range(img1.shape[1]):
            keyPoints[y * img1.shape[1] + x] = cv2.KeyPoint(x,y,size)


    print( 'number of keypoints: ' + str(len(keyPoints)) )
    
    print('calculating sift features...')
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.compute(img1, keyPoints)
    kp2, des2 = sift.compute(img2, keyPoints)

    testPoint = kp1[28 * img1.shape[1] + 57]
    maxDisparity = 100


    for y in range( img1.shape[0] ):
        print('line ' + str(y))
        
        # fit a knn classifier to the row

        for xL in range( img1.shape[1] ):
            if img1[y,xL] == 0:
                continue
            row = y * img1.shape[1]
            offset1 = row + xL
            df1 = des1[offset1]
            kpA = kp1[offset1]
            bestMatchX = -1
            bestMatchDiff = float("inf")


            stopOffset = xL
            startOffset = max(xL-maxDisparity, 0)

            start = y * img1.shape[1] + startOffset
            stop  = y * img1.shape[1] + stopOffset + 1


            nbrs = NearestNeighbors(n_neighbors = 1).fit(des2[start : stop ])
            
            # for each point in the second image, which is closest in the first
            _, ind = nbrs.kneighbors([df1])

            #for xR in range( min(xL,maxDisparity) + 1 ):#x + 1 ):
            #    offset2 = row + xL - xR #- x2 + x
            #    df2 = des2[offset2]
            #    kpB = kp2[offset2]
            #    diff = df2 - df1
            #    dist = diff @ diff

            #    if dist < bestMatchDiff:
            #        bestMatchDiff = dist
            #        bestMatchX = xR #abs(xR-xL)
                    
            #disp = img1.shape[1]
            #if bestMatchX != -1:
            #    disp = bestMatchX
            disp = abs(xL - ind[0,0] - startOffset )

            disparity[y,xL] = disp

    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
    

def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()

#def visualize_disparity_map(disparity, img):
#    plt.imshow(disparity, cmap='jet')
#    plt.imshow(img, alpha = .5)
#    plt.show()

if __name__ == '__main__':
    #if False:
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    #randoms = np.random.choice(pts1.shape[0], pts1.shape[0]//6, replace=False)
    #pts1 = pts1[randoms]
    #pts2 = pts2[randoms]
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 7: generate disparity map
        #cv2.imwrite('leftrect.png',img_left_w)
        #cv2.imwrite('rightrect.png',img_right_w)
    #else:
        #img_left_w = cv2.imread('leftrect.png')
        #img_right_w = cv2.imread('rightrect.png')
        
    scale = 2
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / scale), int(img_left_w.shape[0] / scale)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / scale), int(img_right_w.shape[0] / scale)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    visualize_disparity_map(disparity)

    # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
