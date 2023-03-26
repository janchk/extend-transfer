import uuid

from ExtendTransfer_app.sd_mixer.img_mixer import MixerParams
from ExtendTransfer_app import file_handler


def handler(request):
    session_id = request.COOKIES['id']
    img1_path = file_handler.handle_uploaded_file(request.FILES['image_1'],
                                                  imgtype='img1', id=session_id + '_1')
    img2_path = file_handler.handle_uploaded_file(request.FILES['image_2'],
                                                  imgtype='img2', id=session_id + '_2')

    imgs = []
    imgs.append([float(request.POST['img1_str']), img1_path])
    imgs.append([float(request.POST['img2_str']), img2_path])

    params = MixerParams(
        id=session_id,
        scale=float(request.POST['scale']),
        # scale=5,
        device=request.POST['processing_device'],
        steps=int(request.POST['n_iterations']),
        # steps=10,
        n_samples=1,
        seed=0,
        h=512,
        w=512,
        sampler=request.POST['sampler_method'],
        imgs=imgs 
    )

    return params
