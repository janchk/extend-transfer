from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from .forms import UploadFileForm
from djng_app import file_handler
import uuid


# def index(request):
#     return HttpResponse("Hi, it's django_app index")
def upload_file(request):
    # todo change this function to decorator
    try:
        request.environ['HTTP_REFERER']
    except:
        return redirect('http://127.0.0.1:8080')
    cntimg, simg = 'none', 'none'
    try:
        uniq_id = request.COOKIES['id']
    except:
        uniq_id = str(uuid.uuid4())[:8] # generate uniqe id
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        print('FORM IS VALID =', form.is_valid())
        cntimg = file_handler.handle_uploaded_file(request.FILES['content_img'],
                                                   imgtype='content_image', id=uniq_id)  # send cnt_img
        simg = file_handler.handle_uploaded_file(request.FILES['style_img'], imgtype='style_img', id=uniq_id)  # send style_img
    else:
        form = UploadFileForm()
    response = render(request, 'uploaded.html', {'form': form, 'cimg': cntimg, 'simg': simg})
    response.set_cookie('id', uniq_id, max_age=3600)
    return response