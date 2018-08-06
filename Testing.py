"""
Need testing.
"""
# Handle imports
import unittest

from LoadTilesets import image_to_array, get_tileset_by_id, get_id_of_tileset, get_tileset
from Convert import detect_tileset, check_tileset_all_offsets, get_tile_ids
import numpy as np
import json
import os

# Define constants
test_screenshots = [os.path.join("resources", "screenshots", "Image_Vidumec15x15b.png"),
               os.path.join("resources", "screenshots", "Image_Anikki8x8.png")]
actual_tileset_names = ["Vidumec", "Anikki_square_8x8"]
actual_offsets = [(9, 12), (6, 4)]
actual_tile_id_files = [os.path.join("resources", "test", "Image_Vidumec15x15b_tile_ids"),
               os.path.join("resources", "test", "Image_Anikki8x8_tile_ids")]

# Setup

# Define methods
class TestLoadTilesetMethods(unittest.TestCase):
    def test_image_to_array(self):
        tileset = image_to_array("resources/test/Curses 640x300diag.png")
        assert (tileset == np.load("resources/test/Curses.npy")).all()

class TestConvertMethods(unittest.TestCase):
    def test_detect_tileset0(self):
        self.general_test_tileset(0)

    def test_detect_tileset1(self):
        self.general_test_tileset(1)

    def general_test_tileset(self, i):
        global test_screenshots, actual_tileset_names, actual_offsets

        test_screenshot = test_screenshots[i]
        actual_tileset_name = actual_tileset_names[i]
        actual_offset = actual_offsets[i]

        image = image_to_array(test_screenshot)[:,:,:3]

        tileset_id, offset = detect_tileset(image)

        assert (tileset_id == get_id_of_tileset(actual_tileset_name)), "Detected tileset id: {}, name: {}".format(tileset_id, get_tileset_by_id(tileset_id)["local_filename"])
        assert (tuple(offset) == actual_offset), "Detected offset: {}".format(offset)

    def test_check_tileset_all_offsets0(self):
        self.general_test_check_tileset_all_offsets(0)

    def test_check_tileset_all_offsets1(self):
        self.general_test_check_tileset_all_offsets(1)

    def general_test_check_tileset_all_offsets(self, i):
        global test_screenshots, actual_tileset_names, actual_offsets

        actual_tileset_name = actual_tileset_names[i]
        actual_tileset = get_tileset(actual_tileset_name)
        actual_offset = actual_offsets[i]

        test_screenshot = test_screenshots[i]

        image = image_to_array(test_screenshot)[:, :, :3]

        subset = image[
                        0: 4 * actual_tileset["shape"][0],
                        0: 4 * actual_tileset["shape"][1],
        ]

        _, offset = check_tileset_all_offsets(subset, actual_tileset, [3, 3])

        assert (offset == actual_offset), "Detected offset: {}".format(offset)

    def test_get_tile_ids0(self):
        self.general_test_get_tile_ids(0)

    def test_get_tile_ids1(self):
        self.general_test_get_tile_ids(1)

    def general_test_get_tile_ids(self, i):
        global test_screenshots, actual_tileset_names, actual_offsets, actual_tile_id_files

        actual_tileset_name = actual_tileset_names[i]
        actual_tileset = get_tileset(actual_tileset_name)
        actual_offset = actual_offsets[i]

        with open(actual_tile_id_files[i], "r") as f:
            actual_tile_ids = json.loads(f.read())

        test_screenshot = test_screenshots[i]

        image = image_to_array(test_screenshot)[:, :, :3]

        cropped_image = image[actual_offset[0]:,actual_offset[1]:]

        tile_ids = get_tile_ids(cropped_image, actual_tileset)

        assert (tile_ids.tolist() == actual_tile_ids), "Found tile ids: {}".format(tile_ids.tolist())