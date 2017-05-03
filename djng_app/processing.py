from celery import Celery
from djng_app.style import main
from cstm_clases import Namespace
from django.shortcuts import render, redirect
from celery.result import AsyncResult
from django.http import JsonResponse

def processing(request):
    processed_output = None
    output = 'media/output/output_' + request.COOKIES['id'] + '.jpg'
    try:
        request.POST['on_progress']
    except:
            try:
                images = request.POST['img_to_proceed'].split(',')
            except:
                return render(request, 'Proceeded.html',{'output': output})
            cntimg = images[0]
            simg = images[1]
            args = dict(style_img=simg, content_img=cntimg, gpu_id=-1, model="googlenet", init="content",
                             ratio="1e4", num_iters=2, length=200, verbose=False, output=output)
            processed_output = main.delay(args)
            return JsonResponse({"tsk_id": processed_output.id})
    return JsonResponse({'progr': AsyncResult(request.POST['tsk_id']).result,
                             'sts': AsyncResult(request.POST['tsk_id']).state})
    # if processed_output.state == 'SUCCESS':
        # return render(request, 'Proceeded.html', {'output': output})
    # else:
    #     return JsonResponse({'state': processed_output.state, 'progress': processed_output.result})