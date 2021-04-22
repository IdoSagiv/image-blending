import matplotlib.pyplot as plt
import numpy as np
import imageio
import skimage.color
from scipy.ndimage import filters
import os

MAX_GRAY_LEVEL = 255
RGB_IMAGE = 2
GRAYSCALE_IMAGE = 1
MIN_RESOLUTION = 16
BLUR_FACTOR = 2


def read_image(filename, representation):
    """
    :param filename: image path.
    :param representation: 1=grayscale, 2=RGB.
    :return: a normalized [0,1] image in the requested representation.
    """
    assert filename
    image = imageio.imread(filename).astype(np.float64) / MAX_GRAY_LEVEL
    if representation == GRAYSCALE_IMAGE:
        return skimage.color.rgb2gray(image)
    elif representation == RGB_IMAGE:
        return image


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    :param im: – a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
        in constructing the pyramid filter (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25])
    :return: pyr - the gaussian pyramid as a standard python array (not numpy) with maximum length of max_levels,
                   where each element of the array is a grayscale image
             filter_vec - normalized vector of shape (1, filter_size) built using a consequent 1D convolutions of [1 1]
             with itself in order to derive a row of the binomial coefficients which is a good approximation to the
             Gaussian profile.
    """
    pyr = [im]
    filter_vec = normalize_pascal_coef(filter_size)
    i = 0
    while i < max_levels - 1 and can_reduce(pyr[i]):
        pyr.append(reduce(pyr[i], filter_vec))
        i += 1

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    :param im: – a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
        in constructing the pyramid filter (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25])
    :return: pyr - the laplacian pyramid as a standard python array (not numpy) with maximum length of max_levels,
                   where each element of the array is a grayscale image
             filter_vec - normalized vector of shape (1, filter_size) built using a consequent 1D convolutions of [1 1]
             with itself in order to derive a row of the binomial coefficients which is a good approximation to the
             Gaussian profile.
    """
    gaussian_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    for i in range(len(gaussian_pyr) - 1):
        pyr.append(gaussian_pyr[i] - expand(gaussian_pyr[i + 1], filter_vec))

    pyr.append(gaussian_pyr[-1])
    return pyr, filter_vec


def can_reduce(im):
    """
    :param im:
    :return: true if we can reduce the image and maintain good resolution
    """
    return im.shape[0] >= MIN_RESOLUTION * BLUR_FACTOR and im.shape[1] >= MIN_RESOLUTION * BLUR_FACTOR


def normalize_pascal_coef(n):
    """
    :return: a vector of the n-th pascal coefficient (normalize to sum of 1)
    """
    n_tag = n - 1
    coef = [1]

    for i in range(max(n_tag, 0)):
        coef.append(int(coef[i] * (n_tag - i) / (i + 1)))

    coef = np.array(coef) / np.sum(coef)
    return coef.reshape((1, n))


def blur(im, filter_vec):
    """
    :param im: image to blur
    :param filter_vec: normalized vector of shape (1, filter_size)
    :return: blurred image.
    """
    blur_im = filters.convolve(filters.convolve(im, filter_vec), filter_vec.T)
    return blur_im


def reduce(im, filter_vec):
    """
    blur + sub sample
    :param im: image to reduce
    :param filter_vec: normalized vector of shape (1, filter_size)
    """
    blur_im = blur(im, filter_vec)
    return blur_im[::BLUR_FACTOR, ::BLUR_FACTOR]


def expand(im, filter_vec):
    """
    zero padding + blur
    :param im:
    :param filter_vec:
    """
    new_im = zero_padding(im)
    new_im = blur(new_im, BLUR_FACTOR * filter_vec)
    return new_im


def zero_padding(im):
    """
    pad an image with zeros
    """
    m, n = im.shape
    out = np.zeros((BLUR_FACTOR * m, BLUR_FACTOR * n), dtype=im.dtype)
    out[::BLUR_FACTOR, ::BLUR_FACTOR] = im
    return out


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: the laplacian pyramid as a standard python array (not numpy) with maximum length of max_levels,
            where each element of the array is a grayscale image
    :param filter_vec: normalized vector of shape (1, filter_size) built using a consequent 1D convolutions of [1 1]
            with itself in order to derive a row of the binomial coefficients which is a good approximation to the
            Gaussian profile.
    :param coeff: python list. The list length is the same as the number of levels in the pyramid lpyr. Before
            reconstructing the image img you should multiply each level i of the laplacian pyramid by its corresponding
            coefficient coeff[i].
    :return: the image that represented by the given pyramid.
    """
    n = len(coeff)
    im = lpyr[n - 1] * coeff[n - 1]
    for i in range(n - 2, -1, -1):
        im = expand(im, filter_vec) + (lpyr[i] * coeff[i])
    return im


def linear_stretch(im, min_val, max_val):
    return (im - np.min(im)) * (((max_val - min_val) / (np.max(im) - np.min(im))) + min_val)


def render_pyramid(pyr, levels):
    """

    :param pyr: either a Gaussian or Laplacian pyramid.
    :param levels: the number of levels to present in the result ≤ pyramids max_levels.
    :return: a single black image in which the pyramid levels of the given pyramid pyr are stacked horizontally
        (after stretching the values to [0, 1])
    """
    n = len(pyr[0])
    pyr_im = linear_stretch(pyr[0], 0, 1)
    for i in range(1, levels):
        pad_layer = np.pad(linear_stretch(pyr[i], 0, 1), ((0, n - len(pyr[i])), (0, 0)), 'constant')
        pyr_im = np.hstack((pyr_im, pad_layer))

    return pyr_im


def display_pyramid(pyr, levels):
    """
    display the pyramid as a single black image in which the pyramid levels of the given pyramid pyr are stacked
    horizontally (after stretching the values to [0, 1]).
    :param pyr: either a Gaussian or Laplacian pyramid.
    :param levels: the number of levels to present in the result ≤ pyramids max_levels.
    """
    pyr_im = render_pyramid(pyr, levels)
    plt.imshow(pyr_im, cmap=plt.cm.gray)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Implement pyramid for grayscale images blending as described in the lecture.
    :param im1: first input grayscale image to be blended
    :param im2: second input grayscale image to be blended
    :param mask: a boolean (i.e. dtype == np.bool) mask representing which parts of im1 and im2 should appear in the
        resulting im_blend.
    :param max_levels: the max_levels parameter you should use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im: size of the Gaussian filter which defining the filter used in the construction of the
        Laplacian pyramids of im1 and im2.
    :param filter_size_mask:  size of the Gaussian filter which defining the filter used in the construction of the
        Gaussian pyramid of mask.
    :return: the blended image.
    """
    im1_pyr, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    im2_pyr = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    mask_pyr = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)[0]
    blend_pyr = []
    for i in range(len(im1_pyr)):
        blend_pyr.append(mask_pyr[i] * im1_pyr[i] + (1 - mask_pyr[i]) * im2_pyr[i])
    blend_im = laplacian_to_image(blend_pyr, filter_vec, np.ones(len(blend_pyr)))
    return np.clip(blend_im, 0, 1)


def pyramid_blending_rgb(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Implement pyramid for RGB images blending as described in the lecture.
    :param im1: first input RGB image to be blended
    :param im2: second input RGB image to be blended
    :param mask: a boolean (i.e. dtype == np.bool) mask representing which parts of im1 and im2 should appear in the
        resulting im_blend.
    :param max_levels: the max_levels parameter you should use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im: size of the Gaussian filter which defining the filter used in the construction of the
        Laplacian pyramids of im1 and im2.
    :param filter_size_mask:  size of the Gaussian filter which defining the filter used in the construction of the
        Gaussian pyramid of mask.
    :return: the blended image.
    """
    im1_red, im1_green, im1_blue = im1[:, :, 0], im1[:, :, 1], im1[:, :, 2]
    im2_red, im2_green, im2_blue = im2[:, :, 0], im2[:, :, 1], im2[:, :, 2]
    blend_red = pyramid_blending(im1_red, im2_red, mask, max_levels, filter_size_im, filter_size_mask)
    blend_green = pyramid_blending(im1_green, im2_green, mask, max_levels, filter_size_im, filter_size_mask)
    blend_blue = pyramid_blending(im1_blue, im2_blue, mask, max_levels, filter_size_im, filter_size_mask)

    blend_im = np.dstack((blend_red, blend_green, blend_blue))
    return blend_im


def plot_pyramid_blending(im1, im2, mask, blend_im):
    fig, a = plt.subplots(nrows=2, ncols=2)

    a[0][0].imshow(im1)
    a[0][0].set_title("Image 1")
    a[0][1].imshow(im2)
    a[0][1].set_title("Image 2")
    a[1][0].imshow(mask, cmap='gray')
    a[1][0].set_title("Mask")
    a[1][1].imshow(blend_im)
    a[1][1].set_title("Blended image")

    plt.show()


def images_blending(im1_path, im2_path, mask_path, max_levels, filter_size_im, filter_size_mask):
    """
    performing pyramid blending on two sets of image pairs and masks
    :return: im1, im2= two images
            mask = mask image
            im_blend =the resulting blended image
    """
    im1 = read_image(relpath(im1_path), 2)
    im2 = read_image(relpath(im2_path), 2)
    mask = np.round(read_image(relpath(mask_path), 1)).astype(np.bool)
    im_blend = pyramid_blending_rgb(im1, im2, mask, max_levels, filter_size_im, filter_size_mask)
    plot_pyramid_blending(im1, im2, mask, im_blend)

    return im1, im2, mask, im_blend


def blending_example():
    return images_blending('images/space.jpg', 'images/manhattan.jpg', 'images/mask.jpg', 3, 5, 4)


if __name__ == '__main__':
    blending_example()
