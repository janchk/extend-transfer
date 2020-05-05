from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from .forms import UploadFileForm

from src.djng_app import file_handler
import uuid


def upload_file(request):
    # todo change this function to decorator
    try:
        request.environ['HTTP_REFERER']
    except:
        return redirect('http://127.0.0.1:8080')
    content_img, style_img = 'none', 'none'
    uniq_id = str(uuid.uuid4())[:8]  # generate uniq id per process request
    # try:
    #     uniq_id = request.COOKIES['id']
    # except:
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        print('FORM IS VALID =', form.is_valid())
        content_img = file_handler.handle_uploaded_file(request.FILES['content_img'],
                                                        imgtype='content_img', id=uniq_id)  # send cnt_img
        style_img = file_handler.handle_uploaded_file(request.FILES['style_img'], imgtype='style_img',
                                                      id=uniq_id)  # send style_img
    else:
        form = UploadFileForm()
    response = render(request, 'uploaded.html', {'form': form, 'content_img': content_img, 'style_img': style_img})
    response.set_cookie('id', uniq_id, max_age=3600)
    return response
