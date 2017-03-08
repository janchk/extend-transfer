from django.conf import settings


def handle_uploaded_file(f, imgtype, id):
    # print(imgtype, '!!!!!!!!!!!!!!')
    print('BASEDIIR =', settings.BASE_DIR)
    if imgtype == "content_image":
        imgpth = settings.MEDIA_URL + 'images/content_img/' + id  # for content image
    else:
        imgpth = settings.MEDIA_URL + 'images/style_img/' + id  # for style image
    with open(imgpth, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return imgpth
