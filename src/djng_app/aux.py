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
    content_img_path = os.path.join(media_path, "images/content_img", sess_id)
    style_img_path = os.path.join(media_path, "images/style_img", sess_id)
    return content_img_path, style_img_path
