from django.conf import settings
from django.conf.urls import url
from django.conf.urls.static import static
from django.views.generic import TemplateView

from djng_app.tasks import tskd_processing
from djng_app.views import upload_file

urlpatterns = [
    # url(r'^$', views.index, name='index'),
    url(r'^$', TemplateView.as_view(template_name='file_upload.html'), name='files'),
    url(r'^proceed', tskd_processing, name='proceed'),
    url(r'^upload', upload_file, name='upload_file'),
    # url(r'^pbar', pbar_mng, name='pbar'),
 ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
