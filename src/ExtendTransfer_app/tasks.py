import logging

from ExtendTransfer_app.cstm_clases import Namespace
from ExtendTransfer_app.style_transfer.ImgProcess import ImageProcessor
from ExtendTransfer_proj.celery import app

img_processor = ImageProcessor()
LOG_FORMAT = "%(filename)s:%(funcName)s:%(asctime)s.%(msecs)03d -- %(message)s"


@app.task()
def styletransfer(args):
    args = Namespace(**args)
    level = logging.INFO if args.verbose else logging.DEBUG
    logging.basicConfig(format=LOG_FORMAT, datefmt="%H:%M:%S", level=level)
    logging.info("Starting style transfer.")

    img_processor.process(args)

# #  celery -A ExtendTransfer_proj worker --loglevel=info
