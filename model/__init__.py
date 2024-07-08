from .vision_transformer import VisionTransformerModule

def build_model(args):
    model = VisionTransformerModule()
    return model