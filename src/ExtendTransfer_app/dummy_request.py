from src.ExtendTransfer_app.tasks import styletransfer

if __name__ == "__main__":
    args = dict(content_img="/home/jan/Documents/code/extend-transfer/media/images/content_img/bamboo_forest.jpg",
                style_img="/home/jan/Documents/code/extend-transfer/media/images/style_img/shipwreck.jpg", gpu_id=-1,
                model="vgg16", init="content",
                ratio="1", num_iters=1, length=100, verbose=True,
                output="/home/jan/Documents/code/extend-transfer/media/images/output/8e347556")
    styletransfer(args)
    pass
