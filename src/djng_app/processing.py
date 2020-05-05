from celery import Celery
# from djng_app.style import main
from .cstm_clases import Namespace
from django.shortcuts import render, redirect
from celery.result import AsyncResult
from django.http import JsonResponse
from .tasks import *
from src.djng_app.aux import *


def processing(request):
    processed_output = None
    sess_id = request.COOKIES['id']
    output = 'media/output/output_' + sess_id + '.jpg'  # todo refactor this
    model = "vgg16"  # googlenet, vgg19
    ratio = "1e8"
    num_iters = 1
    length = 200
    try:
        request.POST['on_progress']
    except:
        try:
            images = request.POST['img_to_proceed'].split(',')
        except:
            return render(request, 'proceeded.html', {'output': output, "model": model, "ratio": ratio,
                                                      "num_iters": num_iters,
                                                      "time": AsyncResult(request.POST['tsk_id']).result})
        _content_image_p, _style_image_p = get_imgs_path(sess_id)
        args = dict(style_img=_style_image_p, content_img=_content_image_p, gpu_id=-1, model=model, init="content",
                    ratio=ratio, num_iters=num_iters, length=length, verbose=True, output=output)
        processed_output = styletransfer.delay(args)
        return JsonResponse({"tsk_id": processed_output.id})
    return JsonResponse({'progr': AsyncResult(request.POST['tsk_id']).result,
                         'sts': AsyncResult(request.POST['tsk_id']).state})
    # return JsonResponse({'progr': 50,
    #                      'sts': AsyncResult(request.POST['tsk_id']).state})
