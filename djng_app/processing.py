from djng_app import style
from cstm_clases import Namespace
from django.shortcuts import render, redirect

def processing(request):
    output = 'media/output/output' + '_' + request.COOKIES['id'] + '.jpg'
    try:
        images = request.POST['img_to_proceed'].split(',')
    except:
        return render(request, 'file_upload.html')
    cntimg = images[0]
    simg = images[1]
    args = Namespace(style_img=simg, content_img=cntimg, gpu_id=-1, model='vgg19', init='content',
                     ratio='1e4', num_iters=2, length=200, verbose=False, output=output)
    processed_output = style.main(args)
    print(processed_output)
    return output
