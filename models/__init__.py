from .EAT_pretraining import *
from .finetune_model import *

import fairseq

def build_pretrained_model(args):
    # Load the model, dictionary, and other components
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                filenames=[args.path.pretrained_weights_dir], # gotta be a list due to ensemble expecting multiple model paths
                arg_overrides={'_name':'data2vec_multi'},
                task=fairseq.tasks.audio_finetuning.AudioFinetuningTask(cfg=fairseq.tasks.audio_finetuning.AudioFinetuningConfig)
            )

    # Extract the actual model
    fairseq_model = models[0]

    # Now in order to save some space, we set unimportant contents of the pretraining model to None
    # These features were required during pretraining but not for finetuning.
    fairseq_model.ema = None
    fairseq_model.shared_decoder = None
    fairseq_model.recon_proj = None
    fairseq_model.cls_proj = None
    fairseq_model.center = None

    # Create Additional Linear Classification Layer
    linear_classifier = torch.nn.Linear(in_features=768, out_features=args.dataset.num_classes)

    # Combine into one larger model
    optim_params = {"weight_decay": args.model.weight_decay,"learning_rate":args.model.learning_rate, "n_epochs":args.model.num_epochs}
    model = EATFairseqModule(model=fairseq_model, linear_classifier=linear_classifier, num_classes=args.dataset.num_classes, pos_weight=args.dataset.pos_weight, optim_params=optim_params)
    return model
