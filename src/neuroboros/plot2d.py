import os
import io
import numpy as np
from matplotlib import cm, colors, pyplot as plt, font_manager
from warnings import warn

try:
    from PIL import Image as PIL_Image, ImageDraw, ImageFont
    PIL_ok = True
except ImportError as e:
    PIL_ok = False

try:
    import IPython
    from IPython.display import Image as IPythonImage
    ipython_ok = True
    # ipython_ok = (IPython.get_ipython().__class__.__name__
    #               == 'ZMQInteractiveShell')
except ImportError as e:
    ipython_ok = False

from .io import load_file
from .spaces import get_mapping, get_mask
from .utils import save


PLOT_MAPPING = {}


def unmask_and_upsample(values, space, mask, nn=True):
    ico = int(space.split('-ico')[1])
    nv = ico**2 * 10 + 2

    if mask is not None and mask is not False:
        use_mask = True
        if isinstance(mask, (tuple, list)) and \
                all([isinstance(_, np.ndarray) for _ in mask]):
            masks = mask
        else:
            masks = [get_mask(lr, space) for lr in 'lr']
    else:
        use_mask = False

    if isinstance(values, np.ndarray):
        if use_mask:
            values = np.array_split(values, [masks[0].sum()])
        else:
            values = np.split(values, 2)

    new_values = []
    for v, lr in zip(values, 'lr'):
        if use_mask:
            m = masks['lr'.index(lr)]
            vv = np.full((nv, ) + v.shape[1:], np.nan)
            vv[m] = v
        else:
            vv = v

        mapping = get_mapping(lr, space, 'onavg-ico128', nn=nn)
        vv = mapping.T @ vv

        new_values.append(vv)

    new_values = np.concatenate(new_values, axis=0)
    return new_values


def to_color(values, cmap, vmax=None, vmin=None):
    if vmin is None:
        vmin = -vmax
    r = (values - vmin) / (vmax - vmin)
    r = np.clip(r, 0.0, 1.0)
    cmap = cm.get_cmap(cmap)
    c = cmap(r)
    return c


def prepare_data(
        values, space, mask, nn=True, cmap=None, vmax=None, vmin=None, return_scale=False,
        medial_wall_color=[0.8, 0.8, 0.8, 1.0], background_color=[1.0, 1.0, 1.0, 0.0]):

    values = unmask_and_upsample(values, space, mask, nn=nn)

    if cmap is not None:
        nan_mask = np.isnan(values)
        values = to_color(values, cmap, vmax, vmin)
        values[nan_mask] = medial_wall_color
        values = np.concatenate([values, [_[:values.shape[1]] for _ in [medial_wall_color, background_color]]], axis=0)

        if return_scale:
            norm = colors.Normalize(vmax=vmax, vmin=vmin, clip=True)
            scale = cm.ScalarMappable(norm=norm, cmap=cmap)
            return values, scale

    return values


def brain_plot(values, space, mask, surf_type='inflated', nn=True, cmap=None, vmax=None, vmin=None, return_scale=False,
               medial_wall_color=[0.8, 0.8, 0.8, 1.0], background_color=[1.0, 1.0, 1.0, 0.0],
               colorbar=False, output='ipython', width=None, title=None, title_size=70,
               fn=None, **kwargs):
    assert surf_type in ['inflated', 'pial', 'midthickness', 'white'],\
        f"Surface type '{surf_type}' is not recognized."

    need_scale = return_scale or colorbar

    ret = prepare_data(
        values, space, mask, nn=nn, cmap=cmap, vmax=vmax, vmin=vmin,
        return_scale=need_scale, medial_wall_color=medial_wall_color,
        background_color=background_color)
    if need_scale:
        prepared_values, scale = ret
    else:
        prepared_values = ret

    if surf_type not in PLOT_MAPPING:
        mapping = load_file(os.path.join(
        '2d_plotting_data', f'onavg-ico128_to_{surf_type}_image.npy'))
        PLOT_MAPPING[surf_type] = mapping

    img = prepared_values[PLOT_MAPPING[surf_type]]

    if colorbar:
        if PIL_ok:
            pix_size = (1728, 190)
            dpi = 300
            fig, ax = plt.subplots(1, 1, figsize=[_/dpi for _ in pix_size], dpi=dpi)
            # if 'title' in kwargs:
            #     ax.set_title(kwargs.pop('title'))
            plt.colorbar(scale, shrink=1, aspect=1, cax=ax, orientation='horizontal', **kwargs)
            fig.subplots_adjust(left=0.03, right=0.97, top=0.99, bottom=0.7)
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=dpi, transparent=True)
            buffer.seek(0)
            cbar = PIL_Image.open(buffer)
            plt.close(fig=fig)
        else:
            warn("Cannot convert to Image because Pillow is not installed. "
                 "You can install it with `pip install Pillow`.")

    if output != 'raw':
        img = np.round(img * 255.).astype(np.uint8)
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
                print(img.size, new_img.size, cbar.size)
                new_img.paste(img, (0, offset))
                new_img.paste(cbar, (0, offset+h1))
                img = new_img
            elif title is not None and offset:
                new_img = PIL_Image.new('RGBA', (w1, 50+h1))
                new_img.paste(img, (0, 50))
                img = new_img

            if title is not None:
                font = font_manager.findfont(font_manager.FontProperties())
                font = ImageFont.truetype(font, title_size)
                draw = ImageDraw.Draw(img)
                w, h = draw.textsize(title, font=font)
                x = (img.size[0] - w) / 2
                y = 0
                draw.text((x, y), title, font=font, align='center',
                          fill=(255, 255, 255, 127), stroke_width=3)
                draw.text((x, y), title, font=font, align='center',
                          fill=(0, 0, 0, 255), stroke_width=0)

            if fn is not None:
                save(fn, img)

            if output == 'ipython':
                if ipython_ok:
                    bb = io.BytesIO()
                    img.save(bb, format='png')
                    bb = bb.getvalue()
                    img = IPythonImage(bb, format='png', width=width)
                else:
                    warn("Cannot import `IPython`, skipping conversion to "
                         "`IPython.display.Image`.")
        else:
            warn("Skipping conversion to `PIL.Image` because `Pillow` is not "
                "installed. You can install it with `pip install Pillow`.")

    if return_scale:
        return img, scale
    return img
