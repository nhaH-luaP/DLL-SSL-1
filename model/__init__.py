from .vision_transformer import VisionTransformerModule

def build_model(args):
    model = VisionTransformerModule(image_size_w=args.dataset.width, image_size_h=args.dataset.height, num_classes=args.dataset.num_classes)
    return model