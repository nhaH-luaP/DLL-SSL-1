from .vision_transformer import VisionTransformerModule

def build_model(args):
    model = VisionTransformerModule(image_size_w=args.dataset.width, image_size_h=args.dataset.height, num_classes=args.dataset.num_classes, 
                                    num_layers=args.model.num_layers, num_heads=args.model.num_heads, hidden_dim=args.model.hidden_dim, mlp_dim = args.model.mlp_dim, 
                                    in_channels=args.dataset.in_channels)
    return model