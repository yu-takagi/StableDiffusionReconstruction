import math
from itertools import product

import PIL
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def tile_images(images, ncols=None, columned=False, labels=None, fig=None,
                wspace=0, hspace=0.1, horizontal_margin=0.05, vertical_margin=0.05,
                label_position='inside', label_fontsize=12, label_color='black'):
    '''Create tiled images.

    Parameters
    ----------
    images : list
        List of image files. This can be either
        - List of images.
        - List of lists of images.
    ncols : int
        The number of columns. Default is num of images (i.e., images will be
        tiled in a single row).
    columned : bool
        If `True`, images from different groups are aligned in a column.
        If `False` (default), images are tiled separatedly by groups.
    labels : list
        Labels of image groups. If `None` (default), do not draw label text.
    fig : matplotlib.figure.Figure
        If `None` (default), create the image on `matplotlib.pyplot.gcf()`.
    wspace, hspace : float
    horizontal_margin, vertical_margin : float
    label_position : {'inside' (default)}
        Position of labels (currently only 'inside' is supported).
    label_fontsize : int
        Font size of labels.
    label_color : str
        Label color.

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------

      images = ['/path/to/image0.jpg', '/path/to/image1.jpg', '/path/to/image2.jpg']
      plt.figure()
      tile_images(images)

      +------------------------------+
      | +--------+--------+--------+ |
      | | image0 | image1 | image2 | |
      | +--------+--------+--------+ |
      +------------------------------+

      images = ['/path/to/image0.jpg', '/path/to/image1.jpg', .. '/path/to/image5.jpg']
      plt.figure()
      tile_images(images, ncols=2)

      +---------------------+
      | +--------+--------+ |
      | | image0 | image1 | |
      | +--------+--------+ |
      | | image2 | image3 | |
      | +--------+--------+ |
      | | image4 | image5 | |
      | +--------+--------+ |
      +---------------------+

      images = [['/path/to/A0.jpg', '/path/to/A1.jpg', ..., '/path/to/A5.jpg'],
                ['/path/to/B0.jpg', '/path/to/B1.jpg', ..., '/path/to/B5.jpg'],
      plt.figure()
      tile_images(images, ncols=3)

      +------------------+
      | +----+----+----+ |
      | | A0 | A1 | A2 | |
      | +----+----+----+ |
      | | A3 | A4 | A5 | |
      | +----+----+----+ |
      | +----+----+----+ |
      | | B0 | B1 | B2 | |
      | +----+----+----+ |
      | | B3 | B4 | B5 | |
      | +----+----+----+ |
      +------------------+

      images = [['/path/to/A0.jpg', '/path/to/A1.jpg', ..., '/path/to/A5.jpg'],
                ['/path/to/B0.jpg', '/path/to/B1.jpg', ..., '/path/to/B5.jpg'],
      plt.figure()
      tile_images(images, ncols=3, columned=True)

      +------------------+
      | +----+----+----+ |
      | | A0 | A1 | A2 | |
      | +----+----+----+ |
      | | B0 | B1 | B2 | |
      | +----+----+----+ |
      | +----+----+----+ |
      | | A3 | A4 | A5 | |
      | +----+----+----+ |
      | | B3 | B4 | B5 | |
      | +----+----+----+ |
      +------------------+
    '''

    # Fix `images` to a list of lists
    if not isinstance(images, list):
        images = [[images]]
    else:
        if not isinstance(images[0], list):
            images = [images]

    # Number of images, columns, and rows
    n_img_grp = len(images)
    n_img = max([len(x) for x in images])

    n_global_col = n_img if ncols is None else ncols

    if columned:
        n_global_row = int(math.ceil(n_img / float(n_global_col)))
        n_in_row = n_img_grp
    else:
        n_global_row = n_img_grp
        n_in_row = int(math.ceil(n_img / float(n_global_col)))

    # Open images
    img_obj = []
    imgsize_w_all = []
    imgsize_h_all = []
    for img_g in images:
        img_obj_g = [PIL.Image.open(imgf) for imgf in img_g]
        img_obj.append(img_obj_g)
        imgsize_w_all.extend([img.size[0] for img in img_obj_g])
        imgsize_h_all.extend([img.size[1] for img in img_obj_g])

    imgsize_w = min(imgsize_w_all)
    imgsize_h = min(imgsize_h_all)

    # Figure creation
    if fig is None:
        fig = plt.gcf()

    # Master grid spec
    gs_master = GridSpec(nrows=n_global_row, ncols=1, wspace=wspace, hspace=hspace,
                         left=vertical_margin, right=(1 - vertical_margin),
                         top=(1 - horizontal_margin),
                         bottom=horizontal_margin)

    # Subplots
    for i_global_row in range(n_global_row):
        gs_sub = GridSpecFromSubplotSpec(nrows=1,
                                         ncols=1,
                                         subplot_spec=gs_master[i_global_row, 0],
                                         wspace=0, hspace=0)

        ax = fig.add_subplot(gs_sub[:, :])
        plt.axis('off')

        canvas = PIL.Image.new('RGBA', (imgsize_w * n_global_col, imgsize_h * n_in_row), (0, 0, 0, 0))
        for icol, irow in product(range(n_global_col), range(n_in_row)):
            xpos = icol * imgsize_w
            ypos = irow * imgsize_h

            if columned:
                i_image = i_global_row * n_global_col + icol
                if i_image >= len(img_obj[irow]): continue
                img = img_obj[irow][i_image]
            else:
                i_image = irow * n_global_col + icol
                if i_image >= len(img_obj[i_global_row]): continue
                img = img_obj[i_global_row][i_image]

            img = img.resize((imgsize_w, imgsize_h), PIL.Image.LANCZOS)

            if columned:
                if labels is not None and (icol == 0):
                    txt_x = icol * imgsize_w + (imgsize_w / 20.0)
                    txt_y = irow * imgsize_h + (imgsize_h / 20.0)
                    ax.text(txt_x, txt_y, labels[irow], size=label_fontsize, color=label_color,
                            horizontalalignment='left', verticalalignment='top')
            else:
                if labels is not None and (icol == 0 and irow == 0):
                    txt_x = icol * imgsize_w + (imgsize_w / 20.0)
                    txt_y = irow * imgsize_h + (imgsize_h / 20.0)
                    ax.text(txt_x, txt_y, labels[i_global_row], size=label_fontsize, color=label_color,
                            horizontalalignment='left', verticalalignment='top')

            canvas.paste(img, (xpos, ypos))

        plt.imshow(canvas)

    return fig
