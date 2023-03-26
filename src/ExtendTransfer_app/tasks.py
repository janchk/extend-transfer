import logging

from ExtendTransfer_app.cstm_clases import Namespace
# from ExtendTransfer_app.style_transfer.ImgProcess import ImageProcessor
from ExtendTransfer_app.sd_mixer.img_mixer import ImageMixer
from ExtendTransfer_proj.celery import app

# img_processor = ImageProcessor()
LOG_FORMAT = "%(filename)s:%(funcName)s:%(asctime)s.%(msecs)03d -- %(message)s"
mixer = ImageMixer()

# @app.task()
# def styletransfer(args):
#     args = Namespace(**args)
#     level = logging.INFO if args.verbose else logging.DEBUG
#     logging.basicConfig(format=LOG_FORMAT, datefmt="%H:%M:%S", level=level)
#     logging.info("Starting style transfer.")

#     # img_processor.process(args)

@app.task()
def sd_mix(params):
    # level = logging.INFO if args.verbose else logging.DEBUG
    level = logging.DEBUG
    logging.basicConfig(format=LOG_FORMAT, datefmt="%H:%M:%S", level=level)
    logging.info("Starting mixing process!")
    mixer.process(params)




