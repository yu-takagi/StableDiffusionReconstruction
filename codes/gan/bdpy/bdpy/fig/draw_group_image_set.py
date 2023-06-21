# coding:utf-8

import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageFont
import matplotlib
from matplotlib import font_manager

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = PIL.Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = PIL.Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def draw_group_image_set(condition_list, background_color = (255, 255, 255), 
                         image_size = (160, 160), image_margin = (1, 1, 0, 0), group_margin = (20, 0, 20, 0), max_column_size = 13, 
                         title_fontsize = 20, title_top_padding = 70, title_left_padding = 15, font_family_path = None,
                         id_show = False, id_fontcolor = "black", id_fontsize = 18, image_id_list = [], maintain_aspect_ratio=False,
                         image_padding_color=(0, 0, 0)):
    """
    condition_list : list
        Each condition is a dictionary-type object that contains the following information:
        ```
            condition = {
                "title" : string, # Title name
                "title_fontcolor" :  string or list,   # HTML color name or RGB value list 
                "image_list": list, # The list of image filepath, ndarray or PIL.Image object.  
            }
        ```
        You can also use "image_filepath_list" instead of "image_list".
    background_color : list or tuple
        RGB value list like [Red, Green, Blue].
    image_size: list or tuple
        The image size like [Height, Width].
    image_margin: list or tuple
        The margin of an image like [Top, Right, Bottom, Left].
    group_margin : list or tuple
        The margin of the multiple row images as [Top, Right, Bottom, Left].
    max_column_size : int
        Maximum number of images arranged horizontally.
    title_fontsize : int
        The font size of titles.
    title_top_padding : 
        Top margin of the title letter.
    title_left_padding : 
        Left margin of the title letter.
    font_family_path : string or None
        Font file path.
    id_show : bool
        Specifying whether to display id name.
    id_fontcolor : list or tuple
        Font color of id name.
    id_fontsize : int
        Font size of id name.
    image_id_list : list
        List of id names.
        This list is required when `id_show` is True.
    """

    #------------------------------------
    # Setting
    #------------------------------------

    for condition in condition_list:
        if not condition.get("image_filepath_list") and not condition.get("image_list"):
            raise RuntimeError("The element of `condition_list` needs `image_filepath_list` or `image_list`.")
            return;
        elif condition.get("image_filepath_list") and not condition.get("image_list"):
            condition["image_list"] = condition["image_filepath_list"]

    total_image_size = len(condition_list[0]["image_list"])
    column_size = np.min([max_column_size, total_image_size]) 

    # create canvas
    turn_num = int(np.ceil(total_image_size / float(column_size)))
    nImg_row = len(condition_list) * turn_num 
    nImg_col = 1 + column_size # 1 means title column 
    size_x = (image_size[0] + image_margin[0] + image_margin[2]) * nImg_row + (group_margin[0] + group_margin[2]) * turn_num
    size_y = (image_size[1] + image_margin[1] + image_margin[3]) * nImg_col + (group_margin[1] + group_margin[3])
    image = np.ones([size_x, size_y, 3])
    for bi, bc in enumerate(background_color):
        image[:, :, bi] = bc

    # font settings
    if font_family_path is None:
        font = font_manager.FontProperties(family='sans-serif', weight='normal')
        font_family_path = font_manager.findfont(font)

    #------------------------------------
    # Draw image
    #------------------------------------
    for cind, condition in enumerate(condition_list):
        title = condition['title']
        image_list = condition['image_list']

        for tind in range(total_image_size):
            # Load image
            an_image = image_list[tind]
            if an_image is None: # skip
                continue;
            elif isinstance(an_image, str): # str: filepath
                image_obj = PIL.Image.open(an_image)
            elif isinstance(an_image, np.ndarray): # np.ndarray: array
                image_obj = PIL.Image.fromarray(an_image)
            elif hasattr(an_image, "im"): # im attribute: PIL.Image
                image_obj = an_image
            else:
                raise RuntimeError("What can be treated as an element of `image_list` is only str, ndarray or PIL.Image type.")
                return

            image_obj = image_obj.convert("RGB")
            if maintain_aspect_ratio:
                image_obj = expand2square(image_obj, image_padding_color)
            image_obj = image_obj.resize((image_size[0], image_size[1]), PIL.Image.LANCZOS)

            # Calc image position
            row_index = cind + (tind // column_size) * len(condition_list) 
            column_index = 1 + tind % column_size
            turn_index = tind // column_size       
            x = image_margin[0] + group_margin[0] + row_index * (image_size[0] + image_margin[0] + image_margin[2])
            x += turn_index * (group_margin[0] + group_margin[2])
            y = image_margin[3] + group_margin[3] + column_index * (image_size[1] + image_margin[1] + image_margin[3]) 
            image[ x:(x+image_size[0]), y:(y+image_size[1]), : ] = np.array(image_obj)[:,:,:]

    #------------------------------------
    # Prepare for drawing text
    #------------------------------------
    # cast to unsigned int8
    image = image.astype('uint8')

    # convert ndarray to image object
    image_obj = PIL.Image.fromarray(image)
    draw = PIL.ImageDraw.Draw(image_obj)

    #------------------------------------
    # Draw title name 
    #------------------------------------
    draw.font = PIL.ImageFont.truetype(font=font_family_path, size=title_fontsize)
    for cind, condition in enumerate(condition_list):
        for turn_index in range(turn_num):
            # Calc text position
            row_index = cind + turn_index * len(condition_list) 
            x = image_margin[0] + group_margin[0] + row_index * (image_size[0] + image_margin[0] + image_margin[2])
            x += turn_index * (group_margin[0] + group_margin[2])
            x += title_top_padding 
            y = title_left_padding

            # textの座標指定はxとyが逆転するので注意
            if "title_fontcolor" not in condition.keys():
                title_fontcolor = "black"
            else:
                title_fontcolor = condition["title_fontcolor"]
            draw.text([y, x], condition["title"], title_fontcolor)

    #------------------------------------
    # Draw image id name 
    # * image_id_list variables is necessary
    #------------------------------------

    if id_show:
        draw.font = PIL.ImageFont.truetype(font=font_family_path, size=id_fontsize)
        for tind in range(total_image_size):
            #  Calc text position
            row_index = (tind // column_size) * len(condition_list) 
            column_index = 1 + tind % column_size
            turn_index = tind // column_size            
            x = image_margin[0] + group_margin[0] + row_index * (image_size[0] + image_margin[0] + image_margin[2])
            x += turn_index * (group_margin[0] + group_margin[2])
            x -= id_fontsize
            y = image_margin[3] + group_margin[3] + column_index * (image_size[1] + image_margin[1] + image_margin[3]) 

            draw.text([y, x], image_id_list[tind], id_fontcolor)
            
    return image_obj
