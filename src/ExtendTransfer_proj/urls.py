from django.conf import settings
from django.urls import re_path, path
from django.conf.urls.static import static
from django.views.generic import TemplateView

# from ExtendTransfer_app.processing import processing
from ExtendTransfer_app.views import upload_file, img_mix, mixed

urlpatterns = [
    re_path('$', TemplateView.as_view(
        template_name='file_upload.html'), name='files'),
    re_path(r'^mix$', img_mix, name='mix'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) + \
    static(settings.STATIC_URL, document_root=settings.STATIC_URL)
