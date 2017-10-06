from celery import Celery
# from djng_app.style import main
from cstm_clases import Namespace
from django.shortcuts import render, redirect
from celery.result import AsyncResult
from django.http import JsonResponse
from djng_app.tasks import *

def processing(request):
    processed_output = None
    output = 'media/output/output_' + request.COOKIES['id'] + '.jpg' #todo refactor this
    model = "googlenet" #googlenet, vgg19
    ratio = "1e8"
    num_iters = 10
    length = 200
    try:
        request.POST['on_progress']
    except:
            try:
                images = request.POST['img_to_proceed'].split(',')
            except:
                return render(request, 'proceeded.html',{'output': output, "model": model, "ratio" : ratio,
                                                         "num_iters" : num_iters,
                                                         "time" : AsyncResult(request.POST['tsk_id']).result})
            cntimg,simg = images
            args = dict(style_img=simg, content_img=cntimg, gpu_id=-1, model=model, init="content",
                             ratio=ratio, num_iters=num_iters, length=length, verbose=False, output=output)
            processed_output = styletransfer.delay(args)
            return JsonResponse({"tsk_id": processed_output.id})
    return JsonResponse({'progr': AsyncResult(request.POST['tsk_id']).result,
                             'sts': AsyncResult(request.POST['tsk_id']).state})