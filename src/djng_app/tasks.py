import logging

from src.djng_app.cstm_clases import Namespace
from src.djng_app.style_transfer.ImgProcess import ImageProcessor
from src.django_proj.celery import app

img_processor = ImageProcessor()
LOG_FORMAT = "%(filename)s:%(funcName)s:%(asctime)s.%(msecs)03d -- %(message)s"


@app.task()
def styletransfer(args):
    args = Namespace(**args)
    # level = logging.DEBUG
    level = logging.INFO if args.verbose else logging.DEBUG
    logging.basicConfig(format=LOG_FORMAT, datefmt="%H:%M:%S", level=level)
    logging.info("Starting style transfer.")
    logging.info(args.model)
    logging.info("_______________________________")
    logging.info(args.content_img)

    img_processor.process(args)
    # processed_output = main(args)
    # return processed_output

# #  celery -A django_proj worker --loglevel=info
#
#
# def tskd_processing(request):
#     try:
#         request.environ['HTTP_REFERER']
#     except:
#         return redirect('http://127.0.0.1:8080')
#     tsk = processing(request)
#     #     current_task.update_state()
#     response = render(request, 'proceeded.html', {'out': tsk})
#
#     return response
