from PIL import Image, ImageOps


def fit_contain(image, size):
    return ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
