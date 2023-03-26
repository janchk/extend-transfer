import os

from PIL import Image
from django.conf import settings



def handle_uploaded_file(f, imgtype, id, img_size=None):
    print('BASEDIIR =', settings.BASE_DIR)
    if imgtype == "content_img":
        f_path = settings.MEDIA_ROOT + 'images/content_img'
    elif imgtype == "style_img":
        f_path = settings.MEDIA_ROOT + 'images/style_img'
    elif imgtype == "result_img":
        f_path = settings.MEDIA_ROOT + 'images/mixer/results'
    else:
        f_path = settings.MEDIA_ROOT + 'images/mixer'

    if not os.path.exists(f_path):
        os.makedirs(f_path)
    img_path = os.path.join(f_path, id)

    im = Image.open(f)
    im = im.convert("RGB")
    if img_size:
        im = im.resize((img_size, img_size))
    im.save(img_path, format='JPEG', quality=100)
    return img_path
