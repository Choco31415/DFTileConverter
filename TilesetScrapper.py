"""
This file scrapes tileset images from the DF wiki.
"""
# Define imports
import requests
import json
import re
from datetime import datetime
import time
import urllib
import os
from LoadTilesets import image_to_array
from LoadTilesets import tile_color_guesses
from LoadTilesets import check_image_alpha

# Define constants
wiki_base = "http://dwarffortresswiki.org"
wiki_api = wiki_base + ""
wiki_page = "Tileset_repository"

# Official files
tileset_info_file = os.path.join("resources", "tmp_tileset_info.txt")
tileset_folder = os.path.join("resources", "tilesets")

# Temporary / For editing files
precompute_tileset_info_files = [
    os.path.join("resources", "tmp_tileset_info.txt"),
    os.path.join("resources", "default_tileset_info.txt")
]
postcompute_tileset_info_file = os.path.join("resources", "tileset_info.txt")

api_delay = 0.2 # seconds
last_call = datetime.now().timestamp()

# Define methods
def rate_limit():
    """
    A rate limiting function. If called too frequently, it will delay code execution.

    :return: Nothing
    """
    global last_call

    now = datetime.now().timestamp()
    wait = now - last_call
    if wait < api_delay:
        time.sleep(api_delay - wait)

    last_call = datetime.now().timestamp()

def wiki_get(wiki_api, props):
    """
    A central method for making API calls to a wiki.

    :param wiki_api: The wiki's api.
    :param props: The request props.
    :return: Server text response.
    """
    url = wiki_api+ "/api.php"

    rate_limit()

    return json.loads(requests.get(url, props).text)

def get_wiki_page(wiki_api, pagename):
    """
    Get the wiki markup of a page.

    :param wiki_api: The api location of the wiki.
    :param page: The page to get.
    :return: The Wiki markup of the page.
    """
    props = {"action": "query", "prop": "revisions", "rvprop": "content", "titles": pagename,
             "format": "json"}
    api_response = wiki_get(wiki_api, props)
    page_list = api_response["query"]["pages"]
    page = list(page_list.values())[0]
    return page["revisions"][0]["*"]

def find_tilesets(page_text):
    """
    This method uses a simple heuristic to extract tileset info from a page's text.
    Specifically, it looks at the tileset template.

    :param page_text: The Wiki markup of a page.
    :return: A list of image names.
    """
    global min_tile_size

    # Reformat for easy regex
    page_text = page_text.replace("\n", " ")

    # Get tileset templates
    tilesets = re.findall("\{\{[^{}]*?tileset.*?\}\}", page_text, re.IGNORECASE)

    # Search templates for image files
    tileset_filenames = [re.findall("\[\[[^\]]*((file:|image:)[^\]]*)\]\]|$", snippet, re.IGNORECASE)[0][0] for snippet in tilesets]
    tileset_filenames = [name.split("|")[0].strip() for name in tileset_filenames]

    return tileset_filenames

def download_images(wiki_api, images, local_filenames):
    """
    This method downloads all images from a wiki.

    :param wiki_api: The API location of the wiki.
    :param images: The names of images to download.
    :return: A pinkie party!!
    """
    global batch_size

    direct_urls = []

    for i in range(len(images)):
        image_filename = urllib.parse.unquote(images[i])

        props = {"action": "query", "titles": image_filename, "prop": "imageinfo", "iiprop": "url",
                 "format": "json"}
        api_response = wiki_get(wiki_api, props)
        page = list(api_response["query"]["pages"].values())[0]

        direct_urls.append(page["imageinfo"][0]["url"])

    for i in range(len(images)):
        url = direct_urls[i]
        save_to = local_filenames[i]

        rate_limit()

        urllib.request.urlretrieve(url, save_to)

        print("\rDownloading tileset {}/{}.".format(i, len(images)), end="")

    print("") # Clean up progress bar

def extract_tilesets(wiki_api, pagename):
    """
    This method will find and download tilesets on a wiki page.

    :param wiki: The wiki containing the page.
    :param page: The page to search.
    :return: Nothing
    """
    global tileset_info_file, tileset_folder

    page_text = get_wiki_page(wiki_api, pagename)

    print("Downloaded page.")

    tileset_filenames= find_tilesets(page_text)
    tileset_local_filenames = [filename.split(":")[1] for filename in tileset_filenames]
    tileset_local_filepaths = [os.path.join(tileset_folder, name) for name in tileset_local_filenames]
    download_images(wiki_api, tileset_filenames, tileset_local_filepaths)

    # Save and record some information
    fields = ["wiki_location", "local_filename", "local_filepath"]
    combined_tileset_info = list(zip(tileset_filenames, tileset_local_filenames,
                                     tileset_local_filepaths))
    combined_tileset_info = [dict(zip(fields, tileset)) for tileset in combined_tileset_info]

    print("Extracted tileset info.")

    with open(tileset_info_file, "w+") as f:
        f.write(json.dumps(combined_tileset_info, indent=2))

    print("Finished downloading data.")

def precompute_tilesets(precompute_tileset_info_files, postcompute_tileset_info_file):
    to_return = []

    for file in precompute_tileset_info_files:
        with open(file, "r") as f:
            to_return.extend(json.loads(f.read()))

    print("Precomputing information based on saved data.")

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

    with open(postcompute_tileset_info_file, "w+") as f:
        f.write(json.dumps(to_return, indent=2))

    print("Finished precomputing.")

# Run stuff
extract_tilesets(wiki_api, wiki_page)

precompute_tilesets(precompute_tileset_info_files, postcompute_tileset_info_file)