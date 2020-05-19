import logging

import progressbar as pb
from celery import current_task
from tqdm import tqdm

from ExtendTransfer_app.cstm_clases import Fdout
from ExtendTransfer_app.style_transfer.INetwork import INet
from ExtendTransfer_app.style_transfer.utils import imsave


class ImageProcessor:
    def __init__(self):
        self.path = ""

        # Image size
        # self.image_size = 500
        self.image_size = 300

        # Loss weights
        self.content_weight = 0.025
        self.style_weight = 1.0
        self.style_scale = 1.0
        self.total_variation_weight = 8.5e-5
        self.contet_loss_type = 0

        # Training arguments
        self.num_iterations = 1
        self.model = 'vgg19'
        self.rescale_image = "false"
        self.maintain_aspect_ratio = "false"

        # Transfer Arguments
        self.content_layer = 'conv' + '5_2'
        self.initialization_image = 'content'
        self.pooling_type = 'max'

        # Extra arguments
        self.preserve_color = 'false'
        self.min_improvement = 0.0

    def _create_pbar(self, max_iter):
        """
            Creates a progress bar.
        """

        self.grad_iter = 0
        logging.info(current_task)
        self.pbar = pb.ProgressBar(term_width=0, fd=Fdout(tsk=current_task))
        self.pbar.widgets = [pb.Percentage()]
        # self.pbar.widgets = ["Optimizing: ", pb.Percentage(),
        #                      " ", pb.Bar(marker=pb.AnimatedMarker()),
        #                      " ", pb.ETA()]
        self.pbar.maxval = max_iter

    def process(self, args: dict):
        net_processor = INet()
        net_processor.image_size = args.length
        net_processor.base_image_path = args.content_img
        net_processor.style_image_path = [args.style_img]

        # net_processor.style_scale = ""
        # net_processor.rescale_image = ""
        # net_processor.preserve_color = ""
        # net_processor.init_image = ""  # color
        # net_processor.min_improvement = ""
        net_processor.num_iter = args.num_iters
        # net_processor.content_weight = ""
        # net_processor.style_weight = ""
        net_processor.process()

        self._create_pbar(net_processor.num_iter)
        self.pbar.start()
        for i in tqdm(range(net_processor.num_iter)):
            net_processor.iterate()
        self.pbar.finish()

        img = net_processor.get_result()
        imsave(args.output, img)

    def set_params(self):
        pass
