from .vision_transformer import VisionTransformerModule
from .swin_vision_transformer import SWINVisionTransformerModule

def build_model(args):
    '''
    Some notes about the models:
        - Both models are checked for basic functionality using CIFAR10. However, i removed this part again, as it complicates the pipeline significantly, since the
            lightning data module handels stuff differently then the BirdSetModule. It is however still present (but commented out) in the normal VIT if required again.
        VIT:
            -> I implemented handeling images of non-quadratic from myself, so there may be errors found there if you stumble on problems on a seemingly different but related issue

        SWIN-VIT:
            -> For a better understanding of patch size and window size check out the paper at https://arxiv.org/pdf/2103.14030
    '''
    if args.model.name == 'vit':
        model = VisionTransformerModule(image_size_w=args.dataset.width, image_size_h=args.dataset.height, num_classes=args.dataset.num_classes, 
                                        num_layers=args.model.num_layers, num_heads=args.model.num_heads, hidden_dim=args.model.embed_dim, mlp_dim = args.model.mlp_dim, 
                                        in_channels=args.dataset.in_channels)
    elif args.model.name == 'swin-vit':
        model = SWINVisionTransformerModule(img_size=(args.dataset.height, args.dataset.width), patch_size=args.model.patch_size, in_chans=args.dataset.in_channels, num_classes=args.dataset.num_classes,
                embed_dim=args.model.embed_dim, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=args.model.window_size)
    else:
        raise NotImplementedError('This Model is not implemented!')
    return model