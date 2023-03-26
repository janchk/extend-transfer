from celery.result import AsyncResult
from django.http import JsonResponse
from django.shortcuts import render

from ExtendTransfer_app.aux import *
from ExtendTransfer_app.tasks import styletransfer



def processing(request):
    sess_id = request.COOKIES['id']
    num_iters = int(request.COOKIES['n_iterations'])
    length = int(request.COOKIES['img_size'])
    model = "vgg16"  # googlenet, vgg19
    ratio = "1e8"
    _content_image_p, _style_image_p, _result_image_p = get_imgs_path(sess_id)
    try:
        request.POST['on_progress']
    except:
        try:
            images = request.POST['img_to_process'].split(',')
        except:
            return render(request, 'processed.html', {'output': _result_image_p, "model": model, "ratio": ratio,
                                                      "num_iters": num_iters,
                                                      "time": AsyncResult(request.POST['tsk_id']).result})
        args = dict(style_img=_style_image_p, content_img=_content_image_p, gpu_id=-1, model=model, init="content",
                    ratio=ratio, num_iters=num_iters, length=length, verbose=True, output=_result_image_p)
        processed_output = styletransfer.delay(args)
        return JsonResponse({"tsk_id": processed_output.id})

    print(AsyncResult(request.POST['tsk_id']).result)
    print(AsyncResult(request.POST['tsk_id']).state)

    return JsonResponse({'progr': AsyncResult(request.POST['tsk_id']).result,
                         'sts': AsyncResult(request.POST['tsk_id']).state})
