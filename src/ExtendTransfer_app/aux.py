import os

from django.conf import settings


def get_imgs_path(sess_id: str):
    """

    Args:
        sess_id: ID of request's session

    Returns:
        path to content image, path to style image

    """
    media_path = settings.MEDIA_URL
    content_path = os.path.join(media_path, "images/content_img")
    style_path = os.path.join(media_path, "images/style_img")
    result_path = os.path.join(media_path, "images/result_img")
    for path in (content_path, style_path, result_path):
        if not os.path.exists(path):
            os.makedirs(path)

    content_img_path = os.path.join(content_path, sess_id)
    style_img_path = os.path.join(style_path, sess_id)
    result_img_path = os.path.join(result_path, sess_id + ".jpg")
    return content_img_path, style_img_path, result_img_path
