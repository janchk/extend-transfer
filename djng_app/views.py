from django.shortcuts import render, render_to_response
from django.http import HttpResponseRedirect, HttpResponse
from .forms import UploadFileForm
# from django.core.urlresolvers import reverse
from djng_app import file_handler


def index(request):
    return HttpResponse("Hi, it's django_app index")


def upload_file(request):
    print(request)
    cntimg, simg = 'none', 'none'
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        print('FORM IS VALID =', form.is_valid())
        # if 'content_img' in form.files:
        cntimg = file_handler.handle_uploaded_file(request.FILES['content_img'],
                                                   imgtype='content_image')  # send cnt_img
        # elif 'style_img' in form.files:
        simg = file_handler.handle_uploaded_file(request.FILES['style_img'], imgtype='style_img')  # send style_img
        # else:
        #     HttpResponse('Oh what a pity')
    else:
        form = UploadFileForm()
    # return form.is_valid()
    return render(request, 'uploaded.html', {'form': form, 'cimg': cntimg, 'simg': simg})
    # page after file upload

    # Create your views here.
