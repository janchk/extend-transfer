import subprocess


class ImageProcessor:
    def __init__(self):
        self.path = ""

        # Image size
        self.image_size = 500

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

    def process(self):
        subprocess.run(self.path + "INetwork.py", )

        pass

    def set_params(self):
        pass
