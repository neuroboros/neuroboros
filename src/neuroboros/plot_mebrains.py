import io
import os
from warnings import warn

import numpy as np
from matplotlib import cm, colors, font_manager
from matplotlib import pyplot as plt

try:
    from PIL import Image as PIL_Image
    from PIL import ImageDraw, ImageFont

    PIL_ok = True
except ImportError as e:
    PIL_ok = False

try:
    import IPython
    from IPython.display import Image as IPythonImage
    from IPython.display import display

    ipython_ok = True
    # ipython_ok = (IPython.get_ipython().__class__.__name__
    #               == 'ZMQInteractiveShell')
except ImportError as e:
    ipython_ok = False

from .io import core_dataset
from .spaces import get_mapping, get_mask
from .utils import save

PLOT_MAPPING = {}


def to_color(values, cmap, vmax=None, vmin=None):
    if vmin is None:
        vmin = -vmax
    r = (values - vmin) / (vmax - vmin)
    r = np.clip(r, 0.0, 1.0)
    cmap = cm.get_cmap(cmap)
    c = cmap(r)
    return c


def prepare_data(
    values,
    space,
    mask,
    nn=True,
    cmap=None,
    vmax=None,
    vmin=None,
    return_scale=False,
    medial_wall_color=[0.8, 0.8, 0.8, 1.0],
    background_color=[1.0, 1.0, 1.0, 0.0],
):
    if isinstance(values, np.ndarray):
        pass
    elif isinstance(values, (tuple, list)):
        values = np.concatenate(values, axis=0)
    else:
        raise TypeError(
            "Expected `values` to be a numpy array or a list/"
            f"tuple of numpy arrays. Got {type(values)}."
        )

    if cmap is not None:
        nan_mask = np.isnan(values)
        values = to_color(values, cmap, vmax, vmin)
        values[nan_mask] = medial_wall_color
        values = [
            values,
            [_[: values.shape[1]] for _ in [medial_wall_color, background_color]],
        ]
        values = np.concatenate(values, axis=0)

        if return_scale:
            norm = colors.Normalize(vmax=vmax, vmin=vmin, clip=True)
            scale = cm.ScalarMappable(norm=norm, cmap=cmap)
            return values, scale

    return values


def plot_mebrains(
    values,
    cmap=None,
    vmax=None,
    vmin=None,
    space=None,
    mask=None,
    surf_type='inflated',
    nn=True,
    return_scale=False,
    medial_wall_color=[0.8, 0.8, 0.8, 1.0],
    background_color=[1.0, 1.0, 1.0, 0.0],
    colorbar=True,
    output='ipython',
    width=None,
    title=None,
    title_size=70,
    fn=None,
    **kwargs,
):
    assert surf_type in [
        'inflated',
        'pial',
        'midthickness',
        'white',
    ], f"Surface type '{surf_type}' is not recognized."

    if isinstance(values, np.ndarray):
        cat = values
    elif isinstance(values, (tuple, list)):
        cat = np.concatenate(values, axis=0)
    else:
        raise TypeError(
            "Expected `values` to be a numpy array or a list/"
            f"tuple of numpy arrays. Got {type(values)}."
        )
    ndim = cat.ndim
    percentiles = np.nanpercentile(cat, [5, 95])
    max_, min_ = np.nanmax(cat), np.nanmin(cat)

    if ndim not in [1, 2]:
        raise ValueError(f"Expected `values` to be 1D or 2D. Got {ndim}D.")
    if ndim == 1:
        if cmap is None:
            cmap = 'viridis'
        if vmax is None:
            vmax = percentiles[1]
        if vmin is None:
            vmin = percentiles[0]
    elif ndim == 2:
        if max_ > 1 or min_ < 0:
            raise ValueError("Expected `values` to be in [0, 1] when it's 2D.")
        if cat.shape[1] not in [3, 4]:
            raise ValueError(
                "Expected `values` to have 3 or 4 columns (RGB "
                f"or RGBA). Got {cat.shape[1]} columns."
            )

    need_scale = return_scale or colorbar

    ret = prepare_data(
        values,
        space,
        mask,
        nn=nn,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        return_scale=need_scale,
        medial_wall_color=medial_wall_color,
        background_color=background_color,
    )
    if need_scale:
        prepared_values, scale = ret
    else:
        prepared_values = ret

    if surf_type not in PLOT_MAPPING:
        mapping = core_dataset.get(
            os.path.join(
                '2d_plotting_data_mebrains', f'mebrains_to_{surf_type}_image.npy'
            ),
            on_missing='raise',
        )
        PLOT_MAPPING[surf_type] = mapping

    img = prepared_values[PLOT_MAPPING[surf_type]]

    if colorbar:
        if PIL_ok:
            dpi = 300
            if 'bar_title' in kwargs:
                pix_size = (1728, 250)
                top, bottom = 0.75, 0.55
                fig, ax = plt.subplots(
                    1, 1, figsize=[_ / dpi for _ in pix_size], dpi=dpi
                )
                ax.set_title(kwargs.pop('bar_title'))
            else:
                pix_size = (1728, 190)
                fig, ax = plt.subplots(
                    1, 1, figsize=[_ / dpi for _ in pix_size], dpi=dpi
                )
                top, bottom = 0.99, 0.7
            plt.colorbar(
                scale, shrink=1, aspect=1, cax=ax, orientation='horizontal', **kwargs
            )
            fig.subplots_adjust(left=0.03, right=0.97, top=top, bottom=bottom)
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=dpi, transparent=True)
            buffer.seek(0)
            cbar = PIL_Image.open(buffer)
            plt.close(fig=fig)
        else:
            warn(
                "Cannot convert to Image because Pillow is not installed. "
                "You can install it with `pip install Pillow`."
            )

    if output != 'raw':
        img = np.round(img * 255.0).astype(np.uint8)
    if output in ['ipython', 'pillow'] or fn is not None:
        if PIL_ok:
            img = PIL_Image.fromarray(img)
            if title is not None:
                offset = max(0, title_size - 20)
            else:
                offset = 0

            if colorbar:
                w1, h1 = img.size
                w2, h2 = cbar.size
                new_img = PIL_Image.new('RGBA', (max(w1, w2), h1 + h2))
                # print(img.size, new_img.size, cbar.size)
                new_img.paste(img, (0, offset))
                new_img.paste(cbar, (0, offset + h1))
                img = new_img
            elif title is not None and offset:
                new_img = PIL_Image.new('RGBA', (w1, 50 + h1))
                new_img.paste(img, (0, 50))
                img = new_img

            if title is not None:
                font = font_manager.findfont(font_manager.FontProperties())
                font = ImageFont.truetype(font, title_size)
                draw = ImageDraw.Draw(img)
                w, h = draw.textsize(title, font=font)
                x = (img.size[0] - w) / 2
                y = 0
                draw.text(
                    (x, y),
                    title,
                    font=font,
                    align='center',
                    fill=(255, 255, 255, 127),
                    stroke_width=3,
                )
                draw.text(
                    (x, y),
                    title,
                    font=font,
                    align='center',
                    fill=(0, 0, 0, 255),
                    stroke_width=0,
                )

            if fn is not None:
                save(fn, img)

            if output == 'ipython':
                if ipython_ok:
                    bb = io.BytesIO()
                    img.save(bb, format='png')
                    bb = bb.getvalue()
                    img = IPythonImage(bb, format='png', width=width)
                    display(img)
                    return
                else:
                    warn(
                        "Cannot import `IPython`, skipping conversion to "
                        "`IPython.display.Image`."
                    )
        else:
            warn(
                "Skipping conversion to `PIL.Image` because `Pillow` is not "
                "installed. You can install it with `pip install Pillow`."
            )

    if return_scale:
        return img, scale
    return img
