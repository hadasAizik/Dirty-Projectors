from imageio import imread, imwrite
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import rescale, resize, downscale_local_mean, rotate
from scipy.ndimage import rotate as rotate_3d
from scipy.ndimage import zoom as zoom_3d
from scipy.ndimage import center_of_mass
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import uniform_filter, binary_closing, binary_fill_holes, sobel, binary_dilation, binary_erosion, \
    prewitt, gaussian_gradient_magnitude, map_coordinates
import random
from scipy.spatial.transform import Rotation as R

# Constants
GRAY = 1
RGB = 2
MAX_INTENSITY = 255
PROJECTION_FRONT = 0
PROJECTION_SIDE = 1
PROJECTION_BOTTOM = 2
OPENING = 0
CLOSING = 1
DILATION = 2
EROSION = 3
FILL_HOLES = 4

"""Static functions to deal with pictures - preprocessing etc"""


# ************************************* IMPR functions ***************************************** #

class ImageProcessor:
    """
    @ path - a list of images paths
    @ representation - the pictures color representation
    """

    def __init__(self, folder_path, representation, dimensions):
        """
        Constructor of ImageProcessor.
        @param folder_path: Path of folder containing all images.
        @param representation: Image representation.
        @param dimensions: Tuple (H,W) of target dimension for all images.
        """
        self._representation = representation
        self.images = []
        self.dim = dimensions
        self.load_images(folder_path)
        self.coordiantes = self.get_coordinates()

    def get_coordinates(self):
        face_0 = [0, 0, 0]
        face_1 = [0, 0, 120]
        face_2 = [0, 0, -120]
        face_3 = [0, 90, 0]
        orients = [face_0, face_1, face_2, face_3]
        if len(self.images) != 4:
            return []
        return [get_rotation_coordinates((self.dim[0], self.dim[0], self.dim[0]), orient) for orient in orients]

    def load_images(self, folder_path):
        """
        Load all images from given folder. Perform resizing and quantization.
        @param folder_path: Folder that contains images.
        @return: None.
        """
        self.files = sorted(
            [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
             os.path.isfile(os.path.join(folder_path, f))])
        self.files = list(filter(os.path.exists, self.files))[:]
        for i, file in enumerate(self.files):
            image = self.read_image(file, self._representation)
            image_resized = resize(image, (self.dim[0], self.dim[1]), anti_aliasing=True)

            image_resized = rotate(image_resized, 270)
            image = quantize_image(image_resized)
            self.images.append(image)
        if len(self.images) > 4:
            print('ERROR: the program does not support more than 4 pictures!')
            exit(1)

    def get_image_shape(self):
        """
        @return: Return shape of first image given.
        """
        return self.images[0].shape

    def get_images(self):
        """
        @return: Return given 2d images.
        """
        return self.images

    def read_image(self, filename, representation):
        """
        Reads an image as grayscale or RGB.
        :param filename: path of image file.
        :param representation: 1 for grayscale, 2 for RGB image.
        :return: image matrix.
        """
        image = imread(filename)
        flt_image = image / MAX_INTENSITY
        if representation == GRAY:  # gray
            return rgb2gray(flt_image)
        elif representation == RGB:  # RGB
            return flt_image

    def get_image_projection(self, three_d_matrix, index=0):
        """
        Project the matrix on a specific axis.
        @param three_d_matrix: 3D binary matrix.
        @param axis: axis of projection, TOP, BOTTOM, RIGHT, LEFT, FRONT, BACK.
        @return: Projection of matrix on a given axis.
        """
        if len(self.images) == 4:
            three_d_matrix = map_coordinates(three_d_matrix, self.coordiantes[index], order=1, mode='nearest')
            three_d_matrix = np.rint(three_d_matrix)
            index = 0
        return np.any(three_d_matrix, index).astype(int)

    def loss_2(self, three_d_matrix):
        """
        Calculate loss by l2 norm for all 3-axis projections.
        @param three_d_matrix: 3d binary matrix.
        @return: sum of loss along 3 different axis.
        """
        sum = 0
        for i in range(len(self.images)):
            sum += diff_norm_2(self.images[i], self.get_image_projection(
                three_d_matrix, i))
        return sum

    def loss_1(self, three_d_matrix):
        """
        Calculate loss by l1 norm for all 3-axis projections.
        @param three_d_matrix: 3d binary matrix.
        @return: sum of loss along 3 different axis.
        """
        sum = 0
        for i in range(len(self.images)):
            sum += diff_norm_1(self.images[i], self.get_image_projection(
                three_d_matrix, i))
        return sum

    def loss_penalize_empty_pixels(self, three_d_matrix, alpha=1.4):
        """
        Calculate loss by penalizing outliers and inliers, for all 3-axis projections,
        and reward high volume structures, by penalizing number of empty pixels.
        @param three_d_matrix: 3d binary matrix.
        @return: sum of loss along 3 different axis.
        """
        sum = 0
        three_d_matrix = three_d_matrix
        for i in range(len(self.images)):
            image_projection = self.get_image_projection(three_d_matrix, i)
            sum += 1 * diff_false_inliers(self.images[i], image_projection)
            sum += 4 * diff_false_outliers(self.images[i], image_projection)
        sum += alpha * np.sqrt(count_empty_pixels(three_d_matrix))
        return sum

    def loss_penalize_black_pixels(self, three_d_matrix, alpha=1.4):
        """
        Calculate loss by penalizing outliers and inliers, for all 3-axis projections,
        and reward low volume structures, by penalizing number of black pixels.
        @param three_d_matrix: 3d binary matrix.
        @return: sum of loss along 3 different axis.
        """
        sum = 0
        three_d_matrix = three_d_matrix
        for i in range(len(self.images)):
            image_projection = self.get_image_projection(three_d_matrix, i)
            sum += 1 * diff_false_inliers(self.images[i], image_projection)
            sum += 4 * diff_false_outliers(self.images[i], image_projection)
        sum += alpha * count_black_pixels(three_d_matrix)
        return sum

    def export_result_csv(self, result, file_name):
        """
        Export result to a csv file in order to process later with rhino & GH.
        @param result: 3d binary matrix.
        @param file_name: output csv filename.
        @return: None.
        """
        data = np.where(result > 0)
        # Write the array to disk
        with open(file_name + '.csv', 'w') as outfile:
            # I'm writing a header here just for the sake of readability
            # Any line starting with "#" will be ignored by numpy.loadtxt
            # Iterating through a ndimensional array produces slices along
            # the last axis. This is equivalent to data[i,:,:] in this case
            for data_slice in data:
                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places.
                np.savetxt(outfile, data_slice, delimiter=',', fmt='%-7.2f')

                # Writing out a break to indicate different z slices
                outfile.write('#\n')


def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def quantize_image(image, threshold=0.5):
    """
    Turn image to a 2d black - nan np array.
    :param image: image file.
    :param threshold: all above will turn white, elsewhere black.
    :return: image matrix.
    """
    return np.where(image > threshold, 0, 1)


def quantize_image_reverse(image, threshold=0.5):
    """
    Turn image to a 2d black - white np array.
    :param image: image file.
    :param threshold: all above will turn white, elsewhere black.
    :return: image matrix.
    """
    return np.where(image > threshold, 1, 0)


def diff_norm_1(given_img, projected_img):
    """
    Calculate loss by l1 norm.
    """
    diff = given_img - projected_img
    return np.count_nonzero(diff)


def get_rotation_coordinates(shape, orient):
    phi = orient[0]
    psi = orient[1]
    the = orient[2]

    # create meshgrid
    dim = shape
    ax = np.arange(dim[0])
    ay = np.arange(dim[1])
    az = np.arange(dim[2])
    coords = np.meshgrid(ax, ay, az)

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz = np.vstack([coords[0].reshape(-1) - float(dim[0]) / 2,  # x coordinate, centered
                     coords[1].reshape(-1) - float(dim[1]) / 2,  # y coordinate, centered
                     coords[2].reshape(-1) - float(dim[2]) / 2])  # z coordinate, centered

    # create transformation matrix
    r = R.from_euler('zxz', [phi, psi, the], degrees=True)
    mat = r.as_dcm()

    # apply transformation
    transformed_xyz = np.dot(mat, xyz)

    # extract coordinates
    x = transformed_xyz[0, :] + float(dim[0]) / 2
    y = transformed_xyz[1, :] + float(dim[1]) / 2
    z = transformed_xyz[2, :] + float(dim[2]) / 2

    x = x.reshape((dim[1], dim[0], dim[2]))
    y = y.reshape((dim[1], dim[0], dim[2]))
    z = z.reshape((dim[1], dim[0], dim[2]))  # reason for strange ordering: see next line

    # the coordinate system seems to be strange, it has to be ordered like this
    new_xyz = [y, x, z]

    # return new coordinate system
    return new_xyz


def rotate_array(array, orient):
    phi = orient[0]
    psi = orient[1]
    the = orient[2]

    # create meshgrid
    dim = array.shape
    ax = np.arange(dim[0])
    ay = np.arange(dim[1])
    az = np.arange(dim[2])
    coords = np.meshgrid(ax, ay, az)

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz = np.vstack([coords[0].reshape(-1) - float(dim[0]) / 2,  # x coordinate, centered
                     coords[1].reshape(-1) - float(dim[1]) / 2,  # y coordinate, centered
                     coords[2].reshape(-1) - float(dim[2]) / 2])  # z coordinate, centered

    # create transformation matrix
    r = R.from_euler('zxz', [phi, psi, the], degrees=True)
    mat = r.as_dcm()

    # apply transformation
    transformed_xyz = np.dot(mat, xyz)

    # extract coordinates
    x = transformed_xyz[0, :] + float(dim[0]) / 2
    y = transformed_xyz[1, :] + float(dim[1]) / 2
    z = transformed_xyz[2, :] + float(dim[2]) / 2

    x = x.reshape((dim[1], dim[0], dim[2]))
    y = y.reshape((dim[1], dim[0], dim[2]))
    z = z.reshape((dim[1], dim[0], dim[2]))  # reason for strange ordering: see next line

    # the coordinate system seems to be strange, it has to be ordered like this
    new_xyz = [y, x, z]

    # sample
    return map_coordinates(array, new_xyz, order=1, mode='nearest')


def diff_norm_2(given_img, projected_img):
    """
    Calculate loss by l2 norm.
    """
    diff = given_img - projected_img
    return np.sqrt(np.sum(abs(diff)))


def diff_false_outliers(given_img, projected_img):
    """
    Count number of false outliers, pixels where given_img[pixel] < projected_img[pixel].
    """
    return np.count_nonzero(given_img < projected_img)


def diff_false_inliers(given_img, projected_img):
    """
    Count number of false inliers, pixels where given_img[pixel] > projected_img[pixel].
    """
    return np.count_nonzero(given_img > projected_img)


def resize_3d_matrix(three_d_matrix, length):
    """
    Resize 3d matrix to be in shape of (length, length, length)
    @param three_d_matrix: 3d binary matrix.
    @param length: Length of each dimention.
    @return: Resized 3d matrix.
    """
    return zoom_3d(three_d_matrix, length / three_d_matrix.shape[0], order=0)


def count_empty_pixels(projected_img):
    """
    Count number of empty pixels.
    """
    return np.count_nonzero(projected_img == 0)


def count_black_pixels(projected_img):
    """
    Count number of black pixels.
    """
    return np.count_nonzero(projected_img != 0)


def count_gradient_abs(three_d_image, sigma=10):
    return np.sum(abs(gaussian_gradient_magnitude(three_d_image.astype(float), sigma)))


def diff_center_of_mass_projected(given_img, projected_img):
    """
    Calculate diff of center of mass for every projection separetly.
    @param given_img: Original binary image.
    @param projected_img: Projected binary image.
    @return:
    """
    diff = np.nan_to_num(np.array([center_of_mass(given_img)[i] - center_of_mass(projected_img)[i] for i in range(2)]))
    return np.sum(abs(diff))


def diff_center_of_mass(three_d_matrix):
    """
    Count the distance l1 distance between current matrix center of mass and it's center of volume.
    @param three_d_matrix: 3d binary matrix.
    @return: l1 distance of center of mass and center of volume.
    """
    diff = np.nan_to_num(
        np.array([center_of_mass(three_d_matrix)[i] - (three_d_matrix.shape[0] / 2) for i in range(2)]))
    return np.sum(abs(diff))


def print_3d_matrix(three_d_matrix, rotated_matrix):
    """
    Print two different 3d matrices.
    @return:
    """
    x, y, z = np.where(three_d_matrix == 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='blue')
    x, y, z = np.where(rotated_matrix == 1)
    ax.scatter(x, y, z, c='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.title("resized matrix")
    fig.show()


def merge_matrices(matrix_1, matrix_2, x_start, x_end, y_start, y_end, z_start, z_end):
    new_matrix = np.zeros(matrix_1.shape)
    new_matrix[:, :, :] = matrix_1[:, :, :]
    new_matrix[x_start:x_end, y_start:y_end, z_start:z_end] = matrix_2[x_start:x_end, y_start:y_end, z_start:z_end]
    return new_matrix


def all_neighbouring_indices(matrix_length, x, y, z, neighbour_degree=8):
    neighbouring_indices = []
    if neighbour_degree == 8:
        for sign_x in [-1, 0, 1]:
            for sign_y in [-1, 0, 1]:
                for sign_z in [-1, 0, 1]:
                    if in_boundary(x + sign_x, matrix_length) and in_boundary(y + sign_y,
                                                                              matrix_length) and in_boundary(z + sign_z,
                                                                                                             matrix_length):
                        neighbouring_indices.append((x + sign_x, y + sign_y, z + sign_z))
    else:
        for sign_x in [-1, 1]:
            if in_boundary(x + sign_x, matrix_length):
                neighbouring_indices.append((x + sign_x, y, z))
        for sign_y in [-1, 1]:
            if in_boundary(y + sign_y, matrix_length):
                neighbouring_indices.append((x, y + sign_y, z))
        for sign_z in [-1, 1]:
            if in_boundary(z + sign_z, matrix_length):
                neighbouring_indices.append((x, y, z + sign_z))
    return neighbouring_indices


def in_boundary(coordinate, matrix_length):
    return matrix_length > coordinate >= 0


def binary_operator(three_d_matrix, operator, structure):
    if operator == EROSION:
        return binary_erosion(three_d_matrix, structure=structure)
    elif operator == CLOSING:
        return binary_closing(three_d_matrix, structure=structure)
    elif operator == FILL_HOLES:
        return binary_fill_holes(three_d_matrix)
    else:
        return uniform_filter(three_d_matrix).astype(np.int)


def change_neighbourhood(three_d_matrix, neighbour_flip_prob=0.5):
    """
    select random black pixel in the matrix, and changes the color of pixels in the 3x3 cube
    surrounding that black pixel.
    @param three_d_matrix: 3d binary numpy array.
    @param neighbour_flip_prob: mutation probability for changing a neighoring pixel.
    @return:
    """
    # Select black pixel
    x, y, z = np.where(three_d_matrix == 1)
    if len(x) > 0:
        black_pixel_index = np.random.randint(len(x))
        # Replace block around a currently black pixel.
        new_color = np.random.randint(2)
        neighbouring_indices = all_neighbouring_indices(three_d_matrix.shape[0],
                                                        x[black_pixel_index],
                                                        y[black_pixel_index],
                                                        z[black_pixel_index])
        replace_targets = np.random.choice(2, len(neighbouring_indices),
                                           p=[1 - neighbour_flip_prob, neighbour_flip_prob])
        for j, neighbour_index in enumerate(neighbouring_indices):
            if replace_targets[j]:
                three_d_matrix[neighbour_index] = new_color
    return three_d_matrix


def flip_random_pixels(three_d_matrix, num_pixels=3):
    """Randomly switch the value for num_pixels pixels."""
    dim = three_d_matrix.shape[0]
    random_pixels = [(random.randrange(0, dim), random.randrange(0, dim), random.randrange(0, dim)) for i in
                     range(num_pixels)]
    # switch the dot
    for j in range(num_pixels):
        three_d_matrix[random_pixels[j]] = 1 - three_d_matrix[random_pixels[j]]
    return three_d_matrix


def prob_binary_op(three_d_matrix, length=2):
    """
    deploys random binary operator (erosio, dialation etc, on the matrix)
    @param ind:
    @param length:
    @return:
    """
    operator = np.random.randint(4)
    return binary_operator(three_d_matrix, operator, structure=np.ones((length, length, length))).astype(np.int)


def line_successor(three_d_matrix, neighbour_degree, change_line_prob=0.1):
    blacken_prob = 1 - (count_black_pixels(three_d_matrix) / np.size(three_d_matrix))
    color = np.random.choice(2, 1, p=[1 - blacken_prob, blacken_prob])
    change_line = np.random.choice(2, 1, p=[1 - change_line_prob, change_line_prob])
    x_index, y_index, z_index = np.where(three_d_matrix == 2)
    three_d_matrix[x_index, y_index, z_index] = color
    if change_line or len(x_index) == 0:
        x, y, z = np.where(three_d_matrix == 1)
        black_pixel_index = np.random.randint(len(x))
        # Randomly choose another line start
        x_index = x[black_pixel_index]
        y_index = y[black_pixel_index]
        z_index = z[black_pixel_index]

    # change one neighbour that is not painted yet in your color.
    neighbouring_indices = all_neighbouring_indices(three_d_matrix.shape[0],
                                                    x_index,
                                                    y_index,
                                                    z_index,
                                                    neighbour_degree)
    matrices = [three_d_matrix.copy() for i in range(len(neighbouring_indices))]
    # continue the line
    for i in range(len(neighbouring_indices)):
        if matrices[i][neighbouring_indices[i]] != color:
            matrices[i][neighbouring_indices[i]] = 2
    return matrices
