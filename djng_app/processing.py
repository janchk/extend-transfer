from celery import Celery
from djng_app.style import main
from cstm_clases import Namespace
from django.shortcuts import render, redirect
from celery.result import AsyncResult

# @app.task()
def processing(request):
    output = 'media/output/output' + '_' + request.COOKIES['id'] + '.jpg'
    try:
        images = request.POST['img_to_proceed'].split(',')
    except:
        return render(request, 'file_upload.html')
    cntimg = images[0]
    simg = images[1]
    args = dict(style_img=simg, content_img=cntimg, gpu_id=-1, model="vgg19", init="content",
                     ratio="1e4", num_iters=2, length=200, verbose=False, output=output)
    processed_output = main.delay(args)
    # processed_output = main(args)
    while True:
        print(processed_output.state)
    # processed_output.state
    # result = AsyncResult(id=processed_output)
    print(processed_output)
    return output
