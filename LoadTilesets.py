"""

"""
# Handle imports
from PIL import Image
import numpy as np
import os
import json

# Define constants
tileset_info_file = os.path.join("resources", "tileset_info.txt")
precompute_tileset_info_files = [
    os.path.join("resources", "tmp_tileset_info.txt"),
    os.path.join("resources", "default_tileset_info.txt")
]
tileset_info = None

max_checks = 4 # Max number of color guesses
min_tile_length = 2 # Anything smaller causes subtle bugs in code >.>
max_tile_length = 32 # They're slooowwww
epsilon = 0.001

# Define methods
def load_tileset_info():
    """
    Loads tileset info files and associated images.
    Does some precomputation.

    :param files: Tileset info files
    :return: All the info merged into a list.
    """
    global tileset_info_file, precompute_tileset_info_files, min_tile_length, max_tile_length

    print("Loading in tileset info.")

    recompute_part1 =  False

    if not os.path.exists(tileset_info_file):
        # The tileset info file doesn't exist.
        recompute_part1 = True
    else:
        for file in precompute_tileset_info_files:
            if os.path.getmtime(file) >  os.path.getmtime(tileset_info_file):
                # A file was updated.
                recompute_part1 = True

    if recompute_part1:
        precompute_tilesets_part1(precompute_tileset_info_files, tileset_info_file)

    precompute_tilesets_part2(min_tile_length, max_tile_length, tileset_info_file)

def precompute_tilesets_part1(precompute_tileset_info_files, tileset_info_file):
    """
    Compute non-numpy, easily chacheable data about the tilesets.

    :param precompute_tileset_info_files: Tileset info files from TilesetScrapper.py
    :param tileset_info_file: Where to output computed info to.
    :return: Nothing, really. *sweats*
    """
    to_return = []

    for file in precompute_tileset_info_files:
        with open(file, "r") as f:
            to_return.extend(json.loads(f.read()))

    for info in to_return:
        local_filepath = info["local_filepath"]

        image = image_to_array(local_filepath)
        use_alpha = check_image_alpha(local_filepath)

        tile_shape = [image.shape[0] // 16, image.shape[1] // 16]

        combined_color_guesses = []

        for y in range(16):
            for x in range(16):
                tile = image[
                       y * tile_shape[0]: (y + 1) * tile_shape[0],
                       x * tile_shape[1]: (x + 1) * tile_shape[1]
                       ]

                combined_color_guesses.append(
                    tile_color_guesses(tile, use_alpha))

        info["alpha"] = use_alpha
        info["shape"] = tile_shape
        info["size"] = tile_shape[0] * tile_shape[1]
        info["color_guesses"] = combined_color_guesses

    with open(tileset_info_file, "w+") as f:
        f.write(json.dumps(to_return, indent=2))

    print("Finished precomputing part 1.")

def precompute_tilesets_part2(min_tile_length, max_tile_length, tileset_info_file):
    """
    Compute more information about the tilesets.

    :return: Nothing.
    """
    global tileset_info

    to_return = []

    with open(tileset_info_file, "r") as f:
        to_return.extend(json.loads(f.read()))

    for i in range(len(to_return)-1, -1, -1):
        info = to_return[i]

        image = image_to_array(info["local_filepath"])
        tiles = []

        tile_shape = info["shape"]

        if tile_shape[0] < min_tile_length or tile_shape[1] < min_tile_length:
            del to_return[i]
            continue

        if tile_shape[0] > max_tile_length or tile_shape[1] > max_tile_length:
            del to_return[i]
            continue

        for y in range(16):
            for x in range(16):
                tile = image[
                    y * tile_shape[0]: (y + 1) * tile_shape[0],
                    x * tile_shape[1]: (x + 1) * tile_shape[1]
                             ]
                tile_info = {}
                tile_info["image"] = tile
                tile_info["color_guesses"] = info["color_guesses"][x + y*16]
                tile_info["pink_mask"] = tile_pink_mask(tile)
                tiles.append(tile_info)

        info["tiles"] = np.array(tiles)
        info["hashes"] = calculate_tileset_info(info["tiles"], info["alpha"])

    tileset_info = to_return

    print("Finished precomputing part 2.")

def calculate_tileset_info(tiles, use_alpha):
    """
    Calculate additional info for an entire tileset.

    :param tiles: Tileset tiles
    :param use_alpha: If the tileset uses alpha
    :return: Additional tileset info
    """
    hashes = []

    for i in range(len(tiles)):
        tile = tiles[i]
        hashes.append(hash_tile(tile["image"]))

    return np.array(hashes)

def tile_pink_mask(tile):
    """
    Precompute which parts of a pink tile are background.

    :param tile: A tileset tile
    :return: The tile's pink mask
    """
    # A heuristic
    is_pink = (tile[:, :, 0:1] > 250) * \
              (tile[:, :, 1:2] < 5) * \
              (tile[:, :, 2:3] > 250)
    return is_pink.astype(int)

def hash_tile(tile):
    """
    Quickly calculate a vector of a tile that is color invariant and noise resistant.

    :param tile: A tileset tile
    :return: A numpy vector
    """
    global epsilon

    # Make Hue invariant
    tile = np.sum(tile, axis=2)
    # Make 1D
    tile.resize([tile.shape[0] * tile.shape[1]])

    # Calculate diff between consecutive pixels
    diff = np.convolve(tile, np.array([1, -1]), 'valid')

    np.abs(diff, out=diff)
    diff -= diff.min()
    return diff / (np.linalg.norm(diff) + epsilon)

def entropy_image(image):
    """
    Calculates the entropy of types of colors in an image.

    :param image:
    :return:
    """

    prob = np.sort(
        np.unique(np.resize(image, [-1, image.shape[2]]), return_counts=True,
                  axis=0)[1]) / (
               image.shape[0] * image.shape[1])
    return np.sum(prob * np.log(prob))

def tile_color_guesses(tile, use_alpha):
    """
    Precompute pixels to check for obtaining foreground and background colors of a tile.

    :param tile:
    :param use_alpha:
    :return:
    """
    global max_checks

    # We have to track foreground and background colors for checking purposes
    tile_shape = tile.shape

    checks = []

    foreground_c = False
    background_c = False

    reset = False
    done = False

    # Check each and every pixel
    x = 0
    while x < tile_shape[1] and not done:
        y = 0
        while y < tile_shape[0] and not done:
            tileset_c = tile[y, x]
            alpha = tileset_c[3] / 255
            tileset_c = tileset_c[:3]

            is_pink = tileset_c[0] > 250 and tileset_c[1] < 5 \
                      and tileset_c[2] > 250

            if is_pink and (not use_alpha):
                if not background_c:
                    # The background color and sample color should be the same
                    # Limit to one check, because bleh otherwise
                    checks.append({"type": "p_b", "pos": (y, x)})

                    reset = True # Reset scan with new information
                    background_c = True
            else:
                average = np.max(
                    tileset_c)  # Aka value (HSV) // 255 = fore, 0 = back
                transparency = average / 255

                """
                We need to guess the foreground color.
                If the background color is known, this is easy.
                If alpha = 1.0, we can ignore the background color.
                """
                if alpha == 1 or (
                                background_c is True and alpha != 0.0):

                    if len(checks) < max_checks - 1 or background_c:
                        checks.append({"type": "f", "pos": (y, x)})

                        if not foreground_c:
                            reset = True # Reset scan with new information
                        foreground_c = True

                """
                We need to guess the background color.
                It's pretty much the same as guessing the foreground color.
                """
                if (alpha == 0) or (
                                foreground_c is True and alpha != 1) or (
                            transparency == 0):

                    if len(checks) < max_checks - 1 or foreground_c:
                        checks.append({"type": "b", "pos": (y, x)})

                        if not background_c:
                            reset = True # Reset scan with new information
                        background_c = True

            if (foreground_c and background_c) and len(checks) == max_checks:
                done = True

            y += 1

            if reset:
                y = tile_shape[0]
        x += 1

        if reset:
            x, y = 0, 0
            reset = False

    return checks

def check_image_alpha(image_path):
    """
    Check if an image uses alpha.

    :param image_path:
    :return:
    """
    with Image.open(image_path) as image:
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[0], image.size[1], -1)).astype(int)

    return im_arr.shape[2] == 4

def image_to_array(image_path):
    """
    Loads image into 3D Numpy array of shape
    (width, height, 4)
    Where 4 represents RGBA

    :param image_path: The location of the image
    :return: 3d numpy array
    """
    with Image.open(image_path) as image:
        image = image.convert('RGBA')
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], -1))

    return im_arr

def get_tileset(name):
    """
    Get info for a given tileset.

    :param name: Tileset name
    :return: Tileset info
    """
    global tileset_info

    for info in tileset_info:
        if name.lower() in info["local_filename"].lower():
            return info
    else:
        return None

def get_tileset_by_id(i):
    """
    Get a tileset by an id.

    :param i: An id.
    :return: A corresponding tileset.
    """
    global tileset_info

    return tileset_info[i]

def get_id_of_tileset(name):
    """
    Get an id of a tileset. Returns -1 if nothing matches.

    :param i: An id.
    :return: A corresponding tileset.
    """
    global tileset_info

    id = 0
    for info in tileset_info:
        if name.lower() in info["local_filename"].lower():
            return id
        id += 1
    else:
        return -1

def num_tilesets():
    """
    Get the number of tilesets.

    :return: The number of tilesets
    """
    global tileset_info

    return len(tileset_info)

def largest_tile_dims(tileset_ids = None):
    """
    Get the largest tileset tile dimension.

    :return: A tuple.
    """
    global tileset_info

    max_x = 0
    max_y = 0

    if tileset_ids is None:
        tileset_ids = list(range(num_tilesets()))

    for tileset_id in tileset_ids:
        info = tileset_info[tileset_id]
        if info["shape"][0] > max_y:
            max_y = info["shape"][0]
        if info["shape"][1] > max_x:
            max_x = info["shape"][1]

    largest_dims = [max_y, max_x]

    return largest_dims

smallest_dims = None
def smallest_tile_dims(tileset_ids = None):
    """
    Get the largest tileset tile dimension.

    :return: A tuple.
    """
    global tileset_info

    min_x = 1000
    min_y = 1000

    if tileset_ids is None:
        tileset_ids = list(range(num_tilesets()))

    for tileset_id in tileset_ids:
        info = tileset_info[tileset_id]
        if info["shape"][0] < min_y:
            min_y = info["shape"][0]
        if info["shape"][1] < min_x:
            min_x = info["shape"][1]

    smallest_dims = [min_y, min_x]

    return smallest_dims

# Temp code
# tileset = image_to_array("resources/test/Curses 640x300diag.png")
# np.save('resources/test/Curses', tileset)