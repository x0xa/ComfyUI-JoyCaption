from PIL import Image, ImageOps


def fit_pad_square(image, size):
    image = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    return ImageOps.pad(image, (size, size), color=(0, 0, 0))
