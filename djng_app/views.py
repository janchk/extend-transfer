from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from .forms import UploadFileForm
import logging
from django.core.urlresolvers import reverse
from djng_app import file_handler


def index(request):
    return HttpResponse("Hi, it's django_app index")


def upload_file(request):
    logging.info(request)
    if request.method == 'POST':
        print('RequestMethodIsPOST')
        form = UploadFileForm(request.POST, request.FILES)
        print(form)
        if form.is_valid():
            print('FormIsValid')
            # form.save()
            file_handler.handle_uploaded_file(request.FILES['file'])
            return HttpResponse('hooray!')
    else:
        form = UploadFileForm()
    return render(request, 'uploaded.html', {'form': form})  # страница после загрузки файла
# Create your views here.
