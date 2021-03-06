![An example image conversion, before and after being side by side.](resources/Example.png "Example Conversion")

[![Build Status](https://travis-ci.com/Choco31415/DFTileConverter.svg?branch=master)](https://travis-ci.com/Choco31415/DFTileConverter)

DFTileConverter is a python program that converts a Dwarf Fortress screenshot to a new tileset.

All tilesets in the [DF Wiki tileset repo](http://dwarffortresswiki.org/Tileset_repository) are supported, ignoring those requiring edited raws.

## Setup

`pip install -r requirements.txt`

Note: The initial run will be slower then all subsequent runs.

## Usage

1. Pass in a unique tileset name to `get_tileset` from `LoadTilesets.py`. Here, a unique name refers to a unique string found in the tileset filename.
2. Using the result from step 1, call `convert_screenshot` from `Convert.py`.
3. Celebrate!

Here is an example snippet of code:

```
tileset = get_tileset("cheepicus_15x15.png")
input_filepath = "resources/screenshots/Image_Vidumec15x15a.png"
output_filepath = "resources/tmp/test.png"
convert_screenshot(input_filepath, output_filepath, tileset)
```

## Common Issues

1. The program has obtained tile id's, but is now stuck!
    * This is most likely because the screenshot's tileset is not supported. There are two options:
        1. Please add the tileset to the DF Wiki tileset repository, then re-run `TilesetScrapper.py`.
        2. Manually add the tileset to `resources/default_tileset_info.txt`.

## Project Layout

* Convert
    * This file is aimed at the actual act of converting screenshots between different tilesets.
* LoadTilesets
    * This file is aimed at loading tilesets and interacting with them. Precomputation takes place here.
* Testing
    * This file is aimed at testing parts of the project to ensure stability.
* TilesetScrapper
    * This file is aimed at scrapping tilesets from the DF wiki.

## Contributing

Want to contribute? That's very kind of you!

For documentation purposes, please start by making an issue describing the issue/change, with an example if possible.

Next, make a branch with code changes and submit it via a pull request. The pull request will be run under a few tests, a few of which can sometimes fail.

If everything's looking good, the pull request will be merged in.
