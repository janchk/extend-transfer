from django.shortcuts import render, render_to_response
from django.http import HttpResponseRedirect, HttpResponse
from .forms import UploadFileForm
import logging
# from django.core.urlresolvers import reverse
from djng_app import file_handler


def index(request):
    return HttpResponse("Hi, it's django_app index")


# def show_img(request, arg):
#     render_to_response('uploaded.html', arg)


def upload_file(request):
    print(request)
    cntimg, simg = 'none', 'none'
    if request.method == 'POST':
        # print('RequestMethodIsPOST')
        form = UploadFileForm(request.POST, request.FILES)
        print('FORM IS VALID =', form.is_valid())
        # print(request.POST.pop('imgtype')[0])
        # print('data =', request. )
        # if 'content_img' in form.files:
        cntimg = file_handler.handle_uploaded_file(request.FILES['content_img'], imgtype='content_image')
        # elif 'style_img' in form.files:
        simg = file_handler.handle_uploaded_file(request.FILES['style_img'], imgtype='style_img')
        # else:
        #     HttpResponse('Ой. Какая неожиданнсть.')
    else:
        form = UploadFileForm()
    # return form.is_valid()
    return render(request, 'uploaded.html', {'form': form, 'cimg': cntimg, 'simg': simg})
    # страница после загрузки файла

# Create your views here.