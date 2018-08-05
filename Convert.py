"""

"""
# Handle imports
import numpy as np
from LoadTilesets import get_tileset_by_id, get_id_of_tileset, get_tileset
from LoadTilesets import num_tilesets
from LoadTilesets import image_to_array
from LoadTilesets import largest_tile_dims, smallest_tile_dims
from LoadTilesets import hash_tile
from LoadTilesets import entropy_image
from LoadTilesets import load_tileset_info
from scipy import stats
import scipy.linalg.blas
import scipy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt # DO NOT REMOVE

# Prep
load_tileset_info()

# Define constants
quick_guesses = 1 # Number of image sections to analyze
quick_check_size = 2 # Size of image section to analyze, in tiles
normal_guesses = 1
normal_check_size = 6
normal_tileset_keep = max(int(num_tilesets() * 0.05), 3) # Number of tilesets to keep for a normal check
slow_check_size = 9

# Higher threshold = stricter checks
# Ranges -1 to 1
fast_hash_threshold = 0.97
slow_hash_threshold = 0.8

max_entropy = -0.55 # Entropy required in an analyzed image subset
additional_entropy_check_shapes = [[16, 16], [8, 8]]
max_failed_subsets = 30 # Max number of times to find a suitable subset before requirements are relaxed
kmeans_num_clusters = 18 # Should be 16, but this gives some leeway

epsilon = 0.001

# Testing purposes (probably not used in master)
testing = False
testing2 = False
testing3 = False
testing4 = False

# Define methods
def convert_screenshot(image_path, output_path, new_tileset):
    """
    Convert a screenshot of DF to a new tileset.

    :param image_path: Input screenshot.
    :param output_path: Converted output.
    :param new_tileset: New tileset name to convert to.
    :return: null
    """
    image = image_to_array(image_path)[:,:,:3] # Crop off alpha

    old_tileset_id, tileset_offset = detect_tileset(image)
    old_tileset = get_tileset_by_id(old_tileset_id)

    max_check_size = [image.shape[0] // old_tileset["shape"][0],
                      image.shape[1] // old_tileset["shape"][1]]
    image = image[
            tileset_offset[0]:
            tileset_offset[0] + max_check_size[0] * old_tileset["shape"][0],
            tileset_offset[1]:
            tileset_offset[1] + max_check_size[1] * old_tileset["shape"][1]]

    tile_ids = get_tile_ids(image, old_tileset)

    output_new_map(image, tile_ids, old_tileset, new_tileset, output_path)

def detect_tileset(image):
    """
    Detect the tileset and offset of a screenshot.

    :param image: A screenshot.
    :return: (tileset id, (offset_y, offset_x))
    """
    global quick_guesses, quick_check_size, normal_guesses, normal_check_size, normal_tileset_keep, slow_check_size


    # QUICK CHECK (can make errors)
    print("Quick check.")
    confidence = np.zeros([num_tilesets()])
    offset = []

    for i in range(quick_guesses):
        tmp_confidence, tmp_offset = guess_tileset(image, range(num_tilesets()), quick_check_size)
        confidence += tmp_confidence
        offset.append(tmp_offset)

    # SLOWER CHECK
    print("Normal check.")
    probable_tileset_ids = np.argsort(confidence)[-normal_tileset_keep:]
    confidence = np.zeros([normal_tileset_keep])
    offset = []

    print("Probable tileset ids (worst -> best): {}".format(probable_tileset_ids))

    for i in range(normal_guesses):
        tmp_confidence, tmp_offset = guess_tileset(image, probable_tileset_ids.tolist(), normal_check_size)
        confidence += tmp_confidence
        offset.append(tmp_offset)

    # Find the best tileset.
    max_confidence = np.argwhere(confidence == np.amax(confidence)).flatten()
    count_max_confidence = max_confidence.shape[0]

    if count_max_confidence > 1:
        # SLOWEST CHECK (1 check max
        print("Slow check.")
        confidence, offset = guess_tileset(image, probable_tileset_ids[max_confidence], slow_check_size)

        index_into_probable = max_confidence[np.argmax(confidence)]
        tileset_id = probable_tileset_ids[index_into_probable]
        tileset_offset = offset[np.argmax(confidence)]
    else:
        index_into_probable = np.argmax(confidence)
        tileset_id = probable_tileset_ids[index_into_probable]
        tileset_offset = stats.mode(np.array([o[index_into_probable] for o in offset]), axis=0)[0][0].tolist()

    print()
    print("Detected offset: {}".format(tileset_offset))
    print("Detected tileset: {}".format(tileset_id))

    return tileset_id, tileset_offset

def guess_tileset(image, tileset_ids, check_size):
    """
    For each given tileset, calculate how well it matches a screenshot, and its offset.

    :param image: A screenshot.
    :param tileset_ids: Tilesets to check.
    :param check_size: Tile area of image to analyze.
    :return: (confidence per tileset, (offset_y, offset_x) per tileset)
    """
    global additional_entropy_check_shapes, max_failed_subsets, testing, testing2, testing3, testing4

    confidence = np.zeros([len(tileset_ids)])
    offset = []

    check_size = [check_size, check_size]

    max_check_size = [image.shape[0] // largest_tile_dims(tileset_ids)[0] - 1,
                      image.shape[1] // largest_tile_dims(tileset_ids)[1] - 1]

    if check_size[0] > max_check_size[0]:
        check_size[0] = max_check_size[0]

    if check_size[1] > max_check_size[1]:
        check_size[1] = max_check_size[1]

    entropic = False

    # Make sure we choose a nice subset to run checks on
    check_attempt = 0
    while not entropic:
        entropic = True
        subset_corner = [np.random.randint(
            image.shape[0] - (check_size[0] + 1) * largest_tile_dims(tileset_ids)[0]),
                         np.random.randint(
            image.shape[1] - (check_size[1] + 1) * largest_tile_dims(tileset_ids)[1])]
        subset = image[
                 subset_corner[0]:
                 subset_corner[0] + (check_size[0] + 1) * largest_tile_dims(tileset_ids)[0],
                 subset_corner[1]:
                 subset_corner[1] + (check_size[1] + 1) * largest_tile_dims(tileset_ids)[1]]

        entropy_check_shapes = [smallest_tile_dims(tileset_ids), largest_tile_dims(tileset_ids)] + additional_entropy_check_shapes

        for subset_shape in entropy_check_shapes:
            subset_small = subset[
                           0:(check_size[0] + 1) * subset_shape[0],
                           0:(check_size[1] + 1) * subset_shape[1]]

            entropy = entropy_image(subset_small)

            if entropy >= max_entropy:
                entropic = False

        check_attempt += 1

        if check_attempt == max_failed_subsets:
            # Make subset requirements more lenient
            check_size = [check_size[0] - 1, check_size[1] - 1]

            check_attempt = 0

    print("Subset corner: {}, size: {}".format(subset_corner, check_size))

    for i in range(len(tileset_ids)):
        tileset_id = tileset_ids[i]
        tileset = get_tileset_by_id(tileset_id)
        tile_shape = tileset["shape"]

        tileset_subset = subset[0:(check_size[0] + 1) * tile_shape[0], 0:(check_size[1] + 1) * tile_shape[1]]

        confidence[i], tmp_offset = check_tileset_all_offsets(tileset_subset, tileset, check_size)
        confidence[i] *= tileset["size"] / (check_size[0] * check_size[1])

        tmp_offset = [
            (tmp_offset[0] + subset_corner[0]) % tile_shape[0],
            (tmp_offset[1] + subset_corner[1]) % tile_shape[1]
        ]

        offset.append(tmp_offset)

    return confidence, offset

def check_tileset_all_offsets(subset, tileset, check_size):
    """
    Given a tileset, find the best offset for an image subset, as well as how well it works.

    :param subset: An image subset.
    :param tileset: A tileset to test.
    :param check_size: Tile area of image to analyze.
    :return: confidence, (offset_y, offset_x)
    """
    global max_entropy, testing, testing2, testing3, testing4

    tile_shape = tileset["shape"]

    print("\rChecking tileset: {}".format(tileset["local_filename"]))
    confidence = 0
    offset = (0, 0)

    for offset_x in range(tile_shape[1]):
        for offset_y in range(tile_shape[0]):
            tiles = subset[
                   offset_y : offset_y + check_size[0] * tile_shape[0],
                   offset_x : offset_x + check_size[1] * tile_shape[1]]

            tmp_confidence = check_subset(tiles, tileset, check_size)

            if tmp_confidence > confidence:
                confidence = tmp_confidence
                offset = (offset_y, offset_x)

            print(
                "\rDone with {}/{}".format(offset_y + offset_x * tile_shape[0],
                                           tileset["size"]), end="")

    print("\rTileset confidence: {}".format(confidence))

    return confidence, offset

def check_subset(subset, tileset, check_size):
    """
    Given a tileset, check each subtile to find how well it matches the tileset.

    :param subset: An image subset.
    :param tileset: A tileset.
    :param check_size: Tile area of image to analyze.
    :return: confidence
    """
    global testing, testing2, testing3, testing4

    confidence = 0

    tiles = np.array(np.split(subset, check_size[0], axis=0))
    tiles = np.array(np.split(tiles, check_size[1], axis=2))

    for offset_y in range(check_size[1]):
        for offset_x in range(check_size[0]):
            tile = tiles[offset_y][offset_x]

            tmp_confidence, _ = check_tile(tile, tileset)

            confidence += tmp_confidence

    return confidence

def check_tile(tile, tileset, rendering = False):
    """
    Check how well a tile matches a tileset, and what tile id it could have.

    :param tile: A screenshot tile.
    :param tileset: A tileset.
    :param rendering: Is this method called for image rendering?
    :return: confidence, tile_id
    """
    global testing, testing2, testing3, testing4

    confidence = 0
    best_tileset_id = 0

    tileset_ids = compare_tile_hash(tile, tileset, rendering)
    if rendering:
        if np.max(np.sum(tile, axis=2)) - np.min(np.sum(tile, axis=2)) <= 3:
            tileset_ids = [0] # Eh, lazy

    for tileset_id in tileset_ids:
        tileset_tile = tileset["tiles"][tileset_id]
        use_alpha = tileset["alpha"]

        tmp_confidence = compare_tiles(tile, tileset_tile, use_alpha)

        if tmp_confidence > confidence:
            confidence = tmp_confidence
            best_tileset_id = tileset_id

    return confidence, best_tileset_id

def compare_tile_hash(tile, tileset, rendering = False):
    """
    Compares a tile hash with a tileset.
    This checks what tile id's it might be.

    :param tile: A screenshot tile.
    :param tileset: A tileset.
    :param rendering: Is this method called for image rendering?
    :return: Potential tile id's.
    """
    global fast_hash_threshold, slow_hash_threshold

    hash_threshold = slow_hash_threshold if rendering else fast_hash_threshold

    tile_hash = hash_tile(tile)

    similarity = tileset["hashes"].dot(tile_hash) # Much faster then la.blas.dgemm

    close_tiles = np.nonzero(similarity > hash_threshold)[0]

    return close_tiles

def compare_tiles(tile, tileset_tile, use_alpha):
    """
    Compares a tile with a tileset tile to see how well it matches.

    :param tile: A screenshot tile.
    :param tileset_tile: A tileset tile.
    :param use_alpha: Does the tileset use alpha rendering?
    :return: confidence
    """
    global testing, testing2, testing3, testing4

    tileset_color_guesses = tileset_tile["color_guesses"]
    pink_mask = tileset_tile["pink_mask"]

    foreground_c, background_c = guess_foreground_background(tile, tileset_tile["image"], tileset_color_guesses, use_alpha)

    guess_render = render_color_v(foreground_c, background_c, tileset_tile["image"], pink_mask, use_alpha)

    confidence = 1 - (np.sum(np.abs(guess_render - tile)) / (3 * 255 * tile.shape[0] * tile.shape[1]))

    return confidence

def guess_foreground_background(tile, tileset_tile, tileset_color_guesses, use_alpha):
    """
    Given a tile and a tileset tile, guess the tile's foreground and background colors.
    :param tile: A screenshot tile.
    :param tileset_tile: A tileset tile.
    :param tileset_color_guesses: Pixels of the screenshot tile to check for computing f/b colors.
    :param use_alpha: Does the tileset use alpha rendering?
    :return: foreground, background
    """
    global epsilon, testing, testing2, testing3, testing4
    # We have to track foreground and background colors for checking purposes
    background_c = [] # RGB
    foreground_c = []

    # Check several pixels
    for guess in tileset_color_guesses:
        guess_type = guess["type"]
        y, x = guess["pos"]

        if guess_type == "p_b":
            # The background color and sample color should be the same
            sample_c = tile[y, x].astype(int)

            background_c.append(sample_c)
        else:
            sample_c = tile[y, x].astype(int)
            tileset_c = tileset_tile[y, x].astype(int)

            alpha = (tileset_c[3] + epsilon*0.001) / 255 # Reduce epsilon to avoid weird situations where sample_c is black
            tileset_c = tileset_c[:3]
            transparency = (np.max(tileset_c) + epsilon) / 255

            if guess_type == "f":
                if len(background_c) == 0:
                    background_c2 = np.array([0, 0, 0])
                else:
                    background_c2 = np.average(np.array(background_c), axis=0)

                # Reverse engineer the render code
                boost = (1.0 + (tileset_c - np.max(tileset_c)) / (np.max(tileset_c) + epsilon))
                f_guess = (sample_c - background_c2 * (
                    1 - alpha)) / alpha / transparency
                f_guess /= boost
                f_guess = np.clip(f_guess, 0, 255)

                foreground_c.append(f_guess)

            if guess_type == "b":
                if len(foreground_c) == 0:
                    foreground_c2 = np.array([0, 0, 0])
                else:
                    foreground_c2 = np.average(np.array(foreground_c), axis=0)

                # Reverse engineer the render code
                boost = (tileset_c - np.max(tileset_c)) * (foreground_c2 / (np.max(tileset_c) + epsilon))
                b_guess = (sample_c - (
                    (foreground_c2 + boost) * alpha * transparency)) \
                          / (1 - alpha)
                b_guess = np.clip(b_guess, 0, 255)

                background_c.append(b_guess)

    if len(foreground_c) == 0:
        foreground_c = np.zeros([3])
    else:
        foreground_c = np.average(np.array(foreground_c), axis=0)
    if len(background_c) == 0:
        background_c = np.zeros([3])
    else:
        background_c = np.average(np.array(background_c), axis=0)

    return foreground_c, background_c

def render_color_v(foreground_c, background_c, tileset_tile, pink_mask, use_alpha):
    """
    Given a tileset tile and foreground background colors, render the tile.

    :param foreground_c: The foreground color.
    :param background_c: The background color.
    :param tileset_tile: The tileset tile to render.
    :param pink_mask: The pink mask of the tileset tile.
    :param use_alpha: Does the tileset use alpha rendering?
    :return: A render.
    """
    tileset_tile = tileset_tile.astype(int)

    alpha = tileset_tile[:,:,3:4] / 255
    average = np.max(tileset_tile[:,:,:3], axis=2, keepdims=True)
    transparency = average / 255

    boost = (tileset_tile[:,:,:3] - average) * (foreground_c / (average + epsilon))
    toReturn = (foreground_c + boost) * transparency * alpha + background_c * (1 - alpha)

    if not use_alpha:
        toReturn *= 1 - pink_mask
        toReturn += background_c * np.ones(toReturn.shape) * pink_mask

    toReturn = np.clip(toReturn, 0, 255)

    return toReturn

def get_tile_ids(image, old_tileset):
    """
    Extract the tile ids from a screenshot. The tile grid MUST start at (0,0).

    :param image: A screenshot.
    :param old_tileset: The tileset of the screenshot.
    :return: Tile ids in the image.
    """
    global testing, testing2, testing3, testing4
    tile_shape = old_tileset["shape"]
    check_size = [image.shape[0] // tile_shape[0],
                      image.shape[1] // tile_shape[1]]

    tile_ids = np.zeros(check_size)

    for offset_y in range(check_size[0]):
        for offset_x in range(check_size[1]):

            tile = image[
                   offset_y * tile_shape[0]:
                   (offset_y + 1) * tile_shape[0],
                   offset_x * tile_shape[1]:
                   (offset_x + 1) * tile_shape[1]
                   ]

            _, id = check_tile(tile, old_tileset, rendering=True)
            tile_ids[offset_y, offset_x] = id

    print("Obtained tile id's.")

    return tile_ids

def output_new_map(image, tile_ids,  old_tileset, new_tileset, output_path):
    """
    Convert a screenshot to a new tileset. Requires lots of precomputed info.

    :param image: A screenshot.
    :param tile_ids: Tile ids of tiles in the screenshot.
    :param old_tileset: The old tileset of the screenshot.
    :param new_tileset: The tileset to convert to.
    :param output_path: The path to save the converted screenshot to.
    :return: nothing
    """
    old_tile_shape = old_tileset["shape"]
    new_tile_shape = new_tileset["shape"]

    new_map = np.zeros([len(tile_ids) * new_tile_shape[0], len(tile_ids[0]) * new_tile_shape[1], 3])

    f_colors = np.zeros([len(tile_ids), len(tile_ids[0]), 3])
    b_colors = np.zeros([len(tile_ids), len(tile_ids[0]), 3])

    for offset_y in range(len(tile_ids)):
        for offset_x in range(len(tile_ids[0])):
            tile = image[
                offset_y * old_tile_shape[0]:
                (offset_y + 1) * old_tile_shape[0],
                offset_x * old_tile_shape[1]:
                (offset_x + 1) * old_tile_shape[1]
            ]

            tileset_id = int(tile_ids[offset_y, offset_x])
            old_tileset_tile = old_tileset["tiles"][tileset_id]
            old_tile_color_guesses = old_tileset_tile["color_guesses"]

            f, b = guess_foreground_background(tile, old_tileset_tile["image"], old_tile_color_guesses, old_tileset["alpha"])

            f_colors[offset_y][offset_x], b_colors[offset_y][offset_x] = f, b

    f_colors, b_colors = k_means_cluster(np.array([f_colors, b_colors]))

    for offset_y in range(len(tile_ids)):
        for offset_x in range(len(tile_ids[0])):
            tileset_id = int(tile_ids[offset_y, offset_x])
            new_tileset_tile = new_tileset["tiles"][tileset_id]
            new_pink_mask = new_tileset_tile["pink_mask"]

            f = f_colors[offset_y][offset_x]
            b = b_colors[offset_y][offset_x]

            new_map[
                offset_y * new_tile_shape[0]:
                (offset_y + 1) * new_tile_shape[0],
                offset_x * new_tile_shape[1]:
                (offset_x + 1) * new_tile_shape[1]
            ] = render_color_v(f, b, new_tileset_tile["image"], new_pink_mask,
                               new_tileset["alpha"])

    scipy.misc.imsave(output_path, new_map)

    print("Exported image.")

def k_means_cluster(colors):
    """
    Color recovery has errors.
    Hence, look at colors, cluster them, and pick an average color per cluster.

    :param colors: Colors to cluster.
    :return: Clustered colors.
    """
    global kmeans_num_clusters

    flattened = colors.reshape([-1, colors.shape[-1]])

    kmeans = KMeans(n_clusters=kmeans_num_clusters, random_state=0, n_init = 50).fit(flattened)
    cluster_ids = kmeans.predict(flattened)
    new_colors = kmeans.cluster_centers_[cluster_ids].reshape(colors.shape)

    return new_colors