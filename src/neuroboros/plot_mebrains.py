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
#     from IPython.display import display

#     ipython_ok = True
#     # ipython_ok = (IPython.get_ipython().__class__.__name__
#     #               == 'ZMQInteractiveShell')
# except ImportError as e:
#     ipython_ok = False

from .io import core_dataset
from .plot2d import Image
from .spaces import get_mapping, get_mask
from .utils import save

GUESS_SEPARATE = {
    (10242, 10242): ('mkavg-ico32', False),
    (100988, 100974): ('MEBRAIN', False),
}

GUESS_COMBINED = {
    20484: ('mkavg-ico32', False, [10242]),
    201962: ('MEBRAIN', False, [100988]),
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

    if space != "MEBRAIN":
        ico = int(space.split('-ico')[1])
        nv = ico**2 * 10 + 2

    new_values = []
    for v, lr in zip(values, 'lr'):
        if use_mask:
            m = masks['lr'.index(lr)]
            if space == "MEBRAIN":
                nv = {'l': 100988, 'r': 100974}[lr]
            vv = np.full((nv,) + v.shape[1:], np.nan)
            vv[m] = v
        else:
            vv = v

        if space != "MEBRAIN":
            mapping = get_mapping(lr, space, 'MEBRAIN', nn=nn)
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
            os.path.join(
                '2d_plotting_data_mebrains', f'mebrains_to_{surf_type}_image.npy'
            ),
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
