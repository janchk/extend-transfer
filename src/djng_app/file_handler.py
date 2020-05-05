from django.conf import settings
import os


# todo handle with different types of images
def handle_uploaded_file(f, imgtype, id):
    print('BASEDIIR =', settings.BASE_DIR)
    if imgtype == "content_img":
        f_path = settings.MEDIA_URL + 'images/content_img'
    else:
        f_path = settings.MEDIA_URL + 'images/style_img'

    if not os.path.exists(f_path):
        os.makedirs(f_path)
    img_path = os.path.join(f_path, id)

    with open(img_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return img_path
