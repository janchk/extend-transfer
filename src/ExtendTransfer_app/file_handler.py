import os

from PIL import Image
from django.conf import settings


def handle_uploaded_file(f, imgtype, id, img_size):
    print('BASEDIIR =', settings.BASE_DIR)
    if imgtype == "content_img":
        f_path = settings.MEDIA_URL + 'images/content_img'
    else:
        f_path = settings.MEDIA_URL + 'images/style_img'

    if not os.path.exists(f_path):
        os.makedirs(f_path)
    img_path = os.path.join(f_path, id)

    im = Image.open(f)
    im = im.convert("RGB")
    im = im.resize((img_size, img_size))
    im.save(img_path, format='JPEG', quality=100)
    return img_path
