from io import BytesIO

from PIL import Image


def what(file, h=None):
    try:
        if h is not None:
            data = bytes(h) if isinstance(h, (bytes, bytearray, memoryview)) else h
            image = Image.open(BytesIO(data))
        else:
            image = Image.open(file)
        return image.format.lower() if image.format else None
    except Exception:
        return None
