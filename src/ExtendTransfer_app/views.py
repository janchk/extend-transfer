import uuid

from django.shortcuts import render, redirect

from ExtendTransfer_app import file_handler
from ExtendTransfer_app.forms import UploadFileForm

from io import BytesIO

from PIL import Image


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
        image_size = int(request.POST['image_size'])
        n_iterations = int(request.POST['n_iterations'])
        content_img = file_handler.handle_uploaded_file(request.FILES['content_img'],
                                                        imgtype='content_img', id=uniq_id, img_size=image_size)
        style_img = file_handler.handle_uploaded_file(request.FILES['style_img'], imgtype='style_img',
                                                      id=uniq_id, img_size=image_size)
    else:
        form = UploadFileForm()
    response = render(request, 'uploaded.html', {'form': form, 'content_img': content_img, 'style_img': style_img})
    response.set_cookie('id', uniq_id, max_age=3600)
    response.set_cookie('img_size', image_size, max_age=3600)
    response.set_cookie('n_iterations', n_iterations, max_age=3600)

    return response
