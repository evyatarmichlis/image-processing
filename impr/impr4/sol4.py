# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged 

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.ndimage
from skimage.color import rgb2gray
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
import shutil
from imageio import imwrite
from scipy.ndimage import convolve
import sol4_utils
import imageio
# scipy.ndimage.map_coordinates


VEC_FILTER = np.asarray([1, 0, -1]).reshape(1, 3)

def spread_out_corners(im, m, n, radius):
  """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  corners = [np.empty((0,2), dtype=np.int)]
  x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
  y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
  for i in range(n):
    for j in range(m):
      # Use Harris detector on every sub image.
      sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
      sub_corners = harris_corner_detector(sub_im)
      sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis,:]
      corners.append(sub_corners)
  corners = np.vstack(corners)
  legit = ((corners[:,0]>radius) & (corners[:,0]<im.shape[1]-radius) &
           (corners[:,1]>radius) & (corners[:,1]<im.shape[0]-radius))
  ret = corners[legit,:]
  return ret


def non_maximum_suppression(image):
  """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
  # Find local maximas.
  neighborhood = generate_binary_structure(2,2)
  local_max = maximum_filter(image, footprint=neighborhood)==image
  local_max[image<(image.max()*0.1)] = False

  # Erode areas to single points.
  lbs, num = label(local_max)
  centers = center_of_mass(local_max, lbs, np.arange(num)+1)
  centers = np.stack(centers).round().astype(np.int)
  ret = np.zeros_like(image, dtype=np.bool)
  ret[centers[:,0], centers[:,1]] = True

  return ret




def harris_corner_detector(im):
  """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  #get the der
  Ix = convolve(im,VEC_FILTER)
  Iy = convolve(im,VEC_FILTER.T)
  #blur the image
  M_det,M_trace= det_trace_of_M(Ix,Iy)
  R = M_det - 0.04 * (np.square(M_trace))  # response image
  R_local_max = non_maximum_suppression(R)
  all_true_values = np.argwhere(R_local_max)
  return np.flip(all_true_values, axis=1) # ordered as [column,row]
def det_trace_of_M(Ix,Iy):
  "create matirx for the harris detector"
  Ix_squre_blur = sol4_utils.blur_spatial(Ix*Ix,3)
  Iy_squre_blur = sol4_utils.blur_spatial(Iy*Iy,3)
  Ix_Iy_blur = sol4_utils.blur_spatial(Ix*Iy,3)
  M = np.array([[Ix_squre_blur,Ix_Iy_blur],[Ix_Iy_blur,Iy_squre_blur]])
  M_det = (M[0, 0] * M[1, 1] - (M[0, 1] * M[1, 0]))
  M_trace = M[0, 0] + M[1, 1]
  return M_det,M_trace


def get_range(cols,rows,desc_rad):
  """
  get all the indexes in the radius (r-x+r, r-y+r)
  """

  result = []
  for c,r  in zip(cols,rows):
    rad_of_index = []
    for i in range (-desc_rad,desc_rad+1):
        for j in range (-desc_rad,desc_rad+1) :
          rad_of_index .append([r + i , c + j])
    rad_of_index= np.asarray(rad_of_index)
    result.append(rad_of_index.T)
  return np.asarray(result)


def normalize(d_):
  "normalize the descriptor"
  d = 0
  norm = np.linalg.norm(d_ - np.mean(d_))
  if norm != 0:
    d = np.divide(d_ - np.mean(d_) ,norm)

  return d


def sample_descriptor(im, pos, desc_rad):
  """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image.
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
  """
  k = 2 * desc_rad +1
  cols = pos[:, 0]
  rows = pos[:, 1]
  rad =get_range(cols,rows,desc_rad) # get all the pix index in the radius
  result = np.zeros((len(cols), k, k))
  for i in range(len(cols)):
    d_ = scipy.ndimage.map_coordinates(im, rad[i], order=1, prefilter=False).reshape(k,k)
    d = normalize(d_)
    result[i, :, :] = d
  return result



def find_features(pyr):
  """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
  spread_corners = spread_out_corners(pyr[0],7, 7, 10).astype(np.float64)
  pyr3_corners = (2 ** (- 2)) * spread_corners
  feature_descriptor = sample_descriptor(pyr[2], pyr3_corners , 3)
  return [spread_corners, feature_descriptor]

def find_2nd_max(score):
  "get the 2_max of the cols and the rows"
  score_for_rows = np.copy(score)
  score_for_cols = np.copy(score).T
  first_max_r = np.argmax(score_for_rows,axis=1).reshape(score.shape[0], 1)
  first_max_c = np.argmax(score_for_cols, axis=1).reshape(score.shape[1], 1)
  index_r = np.arange( score.shape[0]).reshape(score.shape[0], 1)
  index_c = np.arange( score.shape[1]).reshape(score.shape[1], 1)
  score_for_rows[index_r, first_max_r] = -np.inf
  score_for_cols[index_c, first_max_c] = -np.inf
  second_max_r = np.argmax(score_for_rows,axis=1).reshape(score.shape[0], 1)
  second_max_c = np.argmax(score_for_cols, axis=1).reshape(score.shape[1], 1)
  score_for_rows[index_r, second_max_r] = np.inf
  score_for_cols[index_c, second_max_c] = np.inf
  score_for_rows[index_r, first_max_r] = np.inf
  score_for_cols[index_c, first_max_c] = np.inf
  return score_for_rows,score_for_cols

def _check_conditions(score_matrix,cols_2nd,rows_2nd,min_score):
  "check the conditions on the score matrix"
  check_min_score = score_matrix >= min_score
  check_2nd_rows = np.invert(score_matrix >= rows_2nd.T)
  check_2nd_cols = np.invert(score_matrix >= cols_2nd)
  good_features_index = check_min_score & check_2nd_rows & check_2nd_cols
  return good_features_index




def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """

    N1, N2, K = desc1.shape[0], desc2.shape[0], desc1.shape[1]
    desc1_N_KK = desc1.reshape(N1, K * K)  # "flatten" the matrix too 2D matrix
    desc2_N_KK = desc2.reshape(N2, K * K)
    score_matrix = desc1_N_KK @ desc2_N_KK.T  # return the Sij matrix N1*N2 all with all the match points
    cols_2nd,rows_2nd = find_2nd_max(score_matrix)
    # the 2 max in the cols and the rows
    good_features_index = _check_conditions(score_matrix,cols_2nd,rows_2nd,min_score)
    inlier_index = np.flip(np.argwhere(good_features_index), axis=1)
    return inlier_index[:, 1], inlier_index[:, 0]


def apply_homography(pos1, H12):
  """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """
  x_y_1 = np.hstack((pos1, np.ones((pos1.shape[0], 1), dtype=pos1.dtype)))
  new_x_y_z = H12.dot(x_y_1.T).T
  z = new_x_y_z[:, -1]
  new_x_y = new_x_y_z[:, :-1]
  normal_x_y = np.divide(new_x_y, z[:, np.newaxis])
  return normal_x_y

def estimate_rigid_transform(points1, points2, translation_only=False):
  """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
  centroid1 = points1.mean(axis=0)
  centroid2 = points2.mean(axis=0)

  if translation_only:
    rotation = np.eye(2)
    translation = centroid2 - centroid1

  else:
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    sigma = centered_points2.T @ centered_points1
    U, _, Vt = np.linalg.svd(sigma)

    rotation = U @ Vt
    translation = -rotation @ centroid1 + centroid2

  H = np.eye(3)
  H[:2,:2] = rotation
  H[:2, 2] = translation
  return H



def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
  """
  Computes homography between two sets of points using RANSAC.
  :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
  """

  best_match = -1

  inliers = None
  inline_index = np.arange(points1.shape[0])
  for i in range(num_iter):
    random_index1, random_index2 = np.random.choice(points1.shape[0], size=2)
    p1,p2 = np.stack((points1[random_index1], points1[random_index2]))\
           ,np.stack((points2[random_index1], points2[random_index2]))
    H12 = estimate_rigid_transform(p1,p2,translation_only) # compute H12 A 3x3 homography matrix
    P2_ = apply_homography(points1,H12)
    E = np.square(np.linalg.norm(P2_-points2,axis=1)) #compute the squared euclidean distance
    check_inlier = E< inlier_tol #Mark all matches having Ej < inlier_tol as inlier matches and the rest as outlier
    curr_match = np.count_nonzero(check_inlier)  #count the number of inliers
    if curr_match > best_match: # update the best match
      best_match = curr_match
      inliers = inline_index[check_inlier] # recompute the inlines

  return [estimate_rigid_transform(points1[inliers, :],points2[inliers, :]
         ,translation_only),inliers.reshape(best_match,)]

def _plot_lines(p1_line,p2_line,color):
  "plot the lines"
  for i in range(p1_line.shape[0]):  # plotting lines
    plt.plot([p1_line[i, 0], p2_line[i, 0]], [p1_line[i, 1], p2_line[i, 1]], mfc='r', c=color, lw=.3, ms=5,marker='.')

def display_matches(im1, im2, points1, points2, inliers):
  """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """

  #get all of the inlines
  p1_inline = points1[inliers] # find the inline
  outline_index1 = np.delete(np.arange(points1.shape[0]), inliers)#find the outlines
  outline_index2 = np.delete(np.arange(points2.shape[0]), inliers)
  p2_inline = points2[inliers]
  p2_inline[:, 0] +=im1.shape[1]
  p1_outline,p2_outline = points1[outline_index1], points2[outline_index2]
  p2_outline[:, 0] +=im1.shape[1]
  image = np.hstack((im1, im2))
  _plot_lines(p1_outline, p2_outline, 'b')
  _plot_lines(p1_inline, p2_inline, 'y')
  plt.imshow(image, cmap='gray')
  plt.show()



def smaller_then_m(m,H_successive):
  "??? For i < m we set Hi,m = Hm???1,m ??? ... ??? Hi+1,i+2 ??? Hi,i+1"
  H2m = [np.eye(3)] # For i = m we set H??i,m to the 3 ?? 3 identity matrix I = np.eye(3)
  for i in reversed(range(m)):
    H = H2m[0] @ H_successive[i - 1]
    H /= H[2, 2]
    H2m.insert(0, H)
  return H2m

def bigger_then_m(m,H_successive ,H2m ):
  " ??? For i > m we set H??i,m = H???1m,m+1 ??? ... ??? H???1i???2,i???1??? H???1i???1,i"
  for i in range(m, len(H_successive)):
    H = H2m[i] @ np.linalg.inv(H_successive[i])
    H /= H[2, 2]
    H2m.append(H)
  return H2m

def accumulate_homographies(H_successive, m):
  """
  Convert a list of succesive homographies to a
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """
  H2m = smaller_then_m(m,H_successive)
  H2m = bigger_then_m(m,H_successive ,H2m )
  return H2m


def compute_bounding_box(homography, w, h):
  """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
  borders = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
  borders_after_homograpy = np.round(apply_homography(borders,H12=homography)) .astype('int')# apply the homography on the corners
  get_x = borders_after_homograpy[:,0]
  get_y = borders_after_homograpy[:,1]
  return np.array([[np.min(get_x),np.min(get_y)],[np.max(get_x),np.max(get_y)]])


def warp_channel(image, homography):
  """
  Warps a 2D image with a given homography.
  :param image: a 2D image.
  :param homography: homography.
  :return: A 2d warped image.
  """

  h, w = image.shape[0], image.shape[1]
  borders = compute_bounding_box(homography, w, h)  # compute the borders of the picture
  range_x = np.arange(borders[0][0],borders[1][0]+1)
  range_y = np.arange(borders[0][1],borders[1][1]+1)
  xv, yv = np.meshgrid(range_x, range_y)  # get the coordinates of the pix
  coordinates = np.stack((xv, yv), axis=-1).reshape(xv.shape[0] * yv.shape[1], 2)
  coordinates_inverse = apply_homography(coordinates, np.linalg.inv(homography)).reshape(xv.shape[0],xv.shape[1],2)  # apply the reverse homography
  warp = (scipy.ndimage.map_coordinates(image, [coordinates_inverse[:,:,1],coordinates_inverse[:,:,0]], order=1, prefilter=False))
  return warp

def warp_image(image, homography):
  """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
  return np.dstack([warp_channel(image[...,channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
  """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
  translation_over_thresh = [0]
  last = homographies[0][0,-1]
  for i in range(1, len(homographies)):
    if homographies[i][0,-1] - last > minimum_right_translation:
      translation_over_thresh.append(i)
      last = homographies[i][0,-1]
  return np.array(translation_over_thresh).astype(np.int)





class PanoramicVideoGenerator:
  """
  Generates panorama from a set of images.
  """

  def __init__(self, data_dir, file_prefix, num_images, bonus=False):
    """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
    self.images = []
    self.bonus = bonus
    self.file_prefix = file_prefix
    self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    self.files = list(filter(os.path.exists, self.files))
    self.panoramas = None
    self.homographies = None
    print('found %d images' % len(self.files))

  def align_images(self, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []

    for file in self.files:
      image = sol4_utils.read_image(file, 1)
      self.images.append(image)
      self.h, self.w = image.shape
      pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
      points_and_descriptors.append(find_features(pyramid))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
      points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
      desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

      # Find matching feature points.
      ind1, ind2 = match_features(desc1, desc2, .7)
      points1, points2 = points1[ind1, :], points2[ind2, :]

      # Compute homography using RANSAC.
      H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

      # Uncomment for debugging: display inliers and outliers among matching points.
      # In the submitted code this function should be commented out!
      # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

      Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    self.homographies = np.stack(accumulated_homographies)
    self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
    self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
      """
      combine slices from input images to panoramas.
      :param number_of_panoramas: how many different slices to take from each input image
      """
      if self.bonus:
        self.generate_panoramic_images_bonus(number_of_panoramas)
      else:
        self.generate_panoramic_images_normal(number_of_panoramas)

  def generate_panoramic_images_normal(self, number_of_panoramas):
    """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
    assert self.homographies is not None

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
    for i in range(self.frames_for_panoramas.size):
      self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(self.bounding_boxes, axis=(0, 1))
    self.bounding_boxes -= global_offset

    slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
      slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
      # homography warps the slice center to the coordinate system of the middle image
      warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
      # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
      warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int)

    self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(self.frames_for_panoramas):
      # warp every input image once, and populate all panoramas
      image = sol4_utils.read_image(self.files[frame_index], 2)
      warped_image = warp_image(image, self.homographies[i])
      x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
      y_bottom = y_offset + warped_image.shape[0]

      for panorama_index in range(number_of_panoramas):
        # take strip of warped image and paste to current panorama
        boundaries = x_strip_boundary[panorama_index, i:i + 2]
        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
        x_end = boundaries[0] + image_strip.shape[1]
        self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    # crop out areas not recorded from enough angles
    # assert will fail if there is overlap in field of view between the left most image and the right most image
    crop_left = int(self.bounding_boxes[0][1, 0])
    crop_right = int(self.bounding_boxes[-1][0, 0])
    assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
    print(crop_left, crop_right)
    self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

  def generate_panoramic_images_bonus(self, number_of_panoramas):
    """
    The bonus
    :param number_of_panoramas: how many different slices to take from each input image
    """
    assert self.homographies is not None

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
    for i in range(self.frames_for_panoramas.size):
      self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(self.bounding_boxes, axis=(0, 1))
    self.bounding_boxes -= global_offset

    slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
      slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
      # homography warps the slice center to the coordinate system of the middle image
      warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
      # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
      warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int)

    self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(self.frames_for_panoramas):
      # warp every input image once, and populate all panoramas
      image = sol4_utils.read_image(self.files[frame_index], 2)
      warped_image = warp_image(image, self.homographies[i])
      x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
      y_bottom = y_offset + warped_image.shape[0]

      for panorama_index in range(number_of_panoramas):
        # take strip of warped image and paste to current panorama
        boundaries = x_strip_boundary[panorama_index, i:i + 2]
        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
        x_end = boundaries[0] + image_strip.shape[1]
        self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip
    # changing to work with our blending functions
    new_pano_x,new_pano_y = 1024 , 1024
    for panorama_index in range(number_of_panoramas):
      # making new padded image
      new_image = np.zeros((new_pano_x, new_pano_y, 3))
      new_image[:panorama_size[1], :panorama_size[0], :] = self.panoramas[panorama_index]

      # zebra_mask
      zebra_mask = np.ones((new_pano_x, new_pano_y))
      for i in range(1, len(x_strip_boundary[panorama_index]) - 1, 2):
        left_boundary = x_strip_boundary[panorama_index][i]
        right_boundary = x_strip_boundary[panorama_index][i + 1]
        zebra_mask[:, left_boundary:right_boundary] = np.zeros((new_pano_x, right_boundary - left_boundary))

      # plt.imshow(zebra_mask, cmap='gray')
      # plt.show()
      blended_image = sol4_utils.blend(new_image, new_image, zebra_mask, 5, 5, 5)

      # back to original size
      self.panoramas[panorama_index] = blended_image[:panorama_size[1], :panorama_size[0], :]

  def save_panoramas_to_video(self):
    assert self.panoramas is not None
    out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
    try:
      shutil.rmtree(out_folder)
    except:
      print('could not remove folder')
      pass
    os.makedirs(out_folder)
    # save individual panorama images to 'tmp_folder_for_panoramic_frames'
    for i, panorama in enumerate(self.panoramas):
      imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
    if os.path.exists('%s.mp4' % self.file_prefix):
      os.remove('%s.mp4' % self.file_prefix)
    # write output video to current folder
    os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
              (out_folder, self.file_prefix))


  def show_panorama(self, panorama_index, figsize=(20, 20)):
    assert self.panoramas is not None
    plt.figure(figsize=figsize)
    plt.imshow(self.panoramas[panorama_index].clip(0, 1))
    plt.show()

  def generate_panoramic_images(self, param):
    """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
    if self.bonus:
      self.generate_panoramic_images_bonus(param)
    else:
      self.generate_panoramic_images_normal(param)


