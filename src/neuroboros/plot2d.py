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

# try:
#     import IPython
#     from IPython.display import Image as IPythonImage
#     from IPython.display import display as IPythonDisplay

#     ipython_ok = True
#     # ipython_ok = (IPython.get_ipython().__class__.__name__
#     #               == 'ZMQInteractiveShell')
# except ImportError as e:
#     ipython_ok = False

from .io import core_dataset
from .spaces import get_mapping, get_mask
from .utils import save

GUESS_SEPARATE = {
    # masked
    (154786, 154560): ('onavg-ico128', True),
    (9675, 9666): ('onavg-ico32', True),
    (9519, 9506): ('fsavg-ico32', True),
    (9504, 9483): ('fslr-ico32', True),
    (38698, 38628): ('onavg-ico64', True),
    (38073, 38040): ('fsavg-ico64', True),
    (38018, 37943): ('fslr-ico64', True),
    (21779, 21731): ('onavg-ico48', True),
    (30153, 30079): ('fslr-ico57', True),
    (152, 151): ('onavg-ico4', True),
    (603, 607): ('onavg-ico8', True),
    (2417, 2414): ('onavg-ico16', True),
    # masked, legacy
    (9372, 9370): ('fsavg-ico32', True, 'fsaverage'),
    # non-masked
    (10242, 10242): ('onavg-ico32', False),
    (40962, 40962): ('onavg-ico64', False),
    (23042, 23042): ('onavg-ico48', False),
    (32492, 32492): ('fslr-ico57', False),
    (162, 162): ('onavg-ico4', False),
    (642, 642): ('onavg-ico8', False),
    (2562, 2562): ('onavg-ico16', False),
}

GUESS_COMBINED = {
    # masked
    309346: ('onavg-ico128', True, [154786]),
    19341: ('onavg-ico32', True, [9675]),
    19025: ('fsavg-ico32', True, [9519]),
    18987: ('fslr-ico32', True, [9504]),
    77326: ('onavg-ico64', True, [38698]),
    76113: ('fsavg-ico64', True, [38073]),
    75961: ('fslr-ico64', True, [38018]),
    43510: ('onavg-ico48', True, [21779]),
    60232: ('fslr-ico57', True, [30153]),
    303: ('onavg-ico4', True, [152]),
    1210: ('onavg-ico8', True, [603]),
    4831: ('onavg-ico16', True, [2417]),
    # masked, legacy
    18742: ('fsavg-ico32', True, [9372], 'fsaverage'),
    # non-masked
    20484: ('onavg-ico32', False, [10242]),
    81924: ('onavg-ico64', False, [40962]),
    46084: ('onavg-ico48', False, [23042]),
    64984: ('fslr-ico57', False, [32492]),
    324: ('onavg-ico4', False, [162]),
    1284: ('onavg-ico8', False, [642]),
    5124: ('onavg-ico16', False, [2562]),
}

PLOT_MAPPING = {}


def unmask_and_upsample(values, space, mask, nn=True):
    if space is None and mask is None:
        if isinstance(values, np.ndarray):
            ret = GUESS_COMBINED[values.shape[0]]
            if len(ret) == 4:
                space, mask, boundary, flavor = ret
                mask_kwargs = {'flavor': flavor, 'legacy': True}
            else:
                space, mask, boundary = ret
                mask_kwargs = {}
        elif isinstance(values, (tuple, list)):
            ret = GUESS_SEPARATE[tuple([_.shape[0] for _ in values])]
            if len(ret) == 3:
                space, mask, flavor = ret
                mask_kwargs = {'flavor': flavor, 'legacy': True}
            else:
                space, mask = ret
                mask_kwargs = {}
        else:
            raise TypeError(
                f"`values` has type `{type(values)}, " "which is not supported."
            )
    else:
        boundary = None

    ico = int(space.split('-ico')[1])
    nv = ico**2 * 10 + 2

    if mask is not None and mask is not False:
        use_mask = True
        if isinstance(mask, (tuple, list)) and all(
            [isinstance(_, np.ndarray) for _ in mask]
        ):
            masks = mask
        else:
            masks = [get_mask(lr, space, **mask_kwargs) for lr in 'lr']
    else:
        use_mask = False

    if isinstance(values, np.ndarray):
        if boundary is not None:
            values = np.array_split(values, boundary)
        elif use_mask:
            values = np.array_split(values, [masks[0].sum()])
        else:
            values = np.split(values, 2)

    new_values = []
    for v, lr in zip(values, 'lr'):
        if use_mask:
            m = masks['lr'.index(lr)]
            vv = np.full((nv,) + v.shape[1:], np.nan)
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
    values = unmask_and_upsample(values, space, mask, nn=nn)

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


def stack_images(images, vertical=True, offset=0):
    widths = [_.size[0] for _ in images]
    heights = [_.size[1] for _ in images]
    if vertical:
        cum_heights = np.cumsum([offset] + heights)
        new_img = PIL_Image.new('RGBA', (max(widths), cum_heights[-1]))
        for img, h in zip(images, cum_heights):
            new_img.paste(img, (0, h))
    else:
        cum_widths = np.cumsum([offset] + widths)
        new_img = PIL_Image.new('RGBA', (cum_widths[-1], max(heights)))
        for img, w in zip(images, cum_widths):
            new_img.paste(img, (w, 0))
    return new_img


class Image:
    def __init__(self, img, scale=None, max_height=500):
        self.img = img
        self.scale = scale
        self.max_height = max_height

    @classmethod
    def stack(cls, images, vertical=True, offset=0):
        images = [_.img if isinstance(_, Image) else _ for _ in images]
        img = stack_images(images, vertical=vertical, offset=offset)
        return cls(img)

    def vstack(self, others, offset=0):
        if isinstance(others, Image):
            others = [others]
        images = [self] + others
        return Image.stack(images, vertical=True, offset=offset)

    def hstack(self, others, offset=0):
        if isinstance(others, Image):
            others = [others]
        images = [self] + others
        return Image.stack(images, vertical=False, offset=offset)

    def title(self, title, size=70):
        font = font_manager.findfont(font_manager.FontProperties())
        font = ImageFont.truetype(font, size)
        draw = ImageDraw.Draw(self.img)
        if hasattr(draw, 'textsize'):
            w, h = draw.textsize(title, font=font)
        else:
            w, h = draw.textbbox((0, 0), text=title, font=font)[2:]
        xy = ((self.img.size[0] - w) / 2, 0)

        title_img = PIL_Image.new('RGBA', (self.img.size[0], h))
        draw = ImageDraw.Draw(title_img)

        draw.text(
            xy,
            title,
            font=font,
            align='center',
            fill=(255, 255, 255, 127),
            stroke_width=3,
        )
        draw.text(
            xy,
            title,
            font=font,
            align='center',
            fill=(0, 0, 0, 255),
            stroke_width=0,
        )
        self.img = stack_images([title_img, self.img], vertical=True)
        return self

    def colorbar(self, scale=None, **kwargs):
        scale = scale if scale is not None else self.scale
        if scale is None:
            raise ValueError("No scale provided for plotting colorbar.")

        dpi = 300
        if 'bar_title' in kwargs:
            pix_size = (1728, 250)
            top, bottom = 0.75, 0.55
            fig, ax = plt.subplots(1, 1, figsize=[_ / dpi for _ in pix_size], dpi=dpi)
            ax.set_title(kwargs.pop('bar_title'))
        else:
            pix_size = (1728, 190)
            fig, ax = plt.subplots(1, 1, figsize=[_ / dpi for _ in pix_size], dpi=dpi)
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
        self.img = stack_images([self.img, cbar], vertical=True)
        return self

    def _repr_png_(self):
        img = self.img
        if self.max_height is not None:
            if self.img.size[1] > self.max_height:
                width = self.max_height * self.img.size[0] // self.img.size[1]
                img = self.img.resize((width, self.max_height))
        bb = io.BytesIO()
        img.save(bb, format='png')
        bb = bb.getvalue()
        return bb
        # Alternatively, this displays the original image with specified width
        # img = IPythonImage(bb, format='png', width=width)
        # IPythonDisplay(img)

    def save(self, fn):
        save(fn, self.img)


def brain_plot(
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
    output=None,
    width=500,
    title=None,
    title_size=70,
    fn=None,
    **kwargs,
):
    if output is None and fn is None:
        output = 'pillow'

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
    percentiles = np.nanpercentile(cat, [1, 99])
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
            os.path.join('2d_plotting_data', f'onavg-ico128_to_{surf_type}_image.npy'),
            on_missing='raise',
        )
        PLOT_MAPPING[surf_type] = mapping

    img = prepared_values[PLOT_MAPPING[surf_type]]

    if output == 'raw':
        if return_scale:
            return img, scale
        return img

    img = np.round(img * 255.0).astype(np.uint8)
    if not PIL_ok:
        warn(
            "Skipping conversion to `PIL.Image` because `Pillow` is not "
            "installed. You can install it with `pip install Pillow`."
        )
        if return_scale:
            return img, scale
        return img

    img = PIL_Image.fromarray(img)
    img = Image(img)
    if title is not None:
        img = img.title(title, size=title_size)
    if colorbar:
        img = img.colorbar(scale, **kwargs)

    if fn is not None:
        img.save(fn)

    if return_scale:
        return img, scale
    return img
