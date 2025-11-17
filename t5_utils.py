import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# ===============================
#  WandB setup
# ===============================
def setup_wandb(args):
    if args.use_wandb:
        wandb.init(project="t5_sql_project", name=args.experiment_name, config=vars(args))


# ===============================
#  Model initialization
# ===============================
def initialize_model(args):
    """
    Initializes the T5 model and tokenizer.
    Uses pretrained weights if --finetune is set,
    else loads model config for training from scratch.
    """
    model_name = "google-t5/t5-small"
    print("\n========== Initializing model ==========")
    
    if args.finetune:
        print("🔄 Loading pretrained model:", model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        print("🧱 Initializing model from config (training from scratch)")
        config = transformers.T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(config)

    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model.to(DEVICE)

    print(f"✅ Model and tokenizer loaded on {DEVICE}")
    return model, tokenizer


# ===============================
#  Directory management
# ===============================
def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass


# ===============================
#  Model saving
# ===============================
def save_model(checkpoint_dir, model, best):
    mkdir(checkpoint_dir)
    tag = "best" if best else "latest"
    save_path = os.path.join(checkpoint_dir, f"t5_{tag}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Saved {tag} model to {save_path}")


# ===============================
#  Model loading
# ===============================
def load_model_from_checkpoint(args, best=True):
    model_name = "google-t5/t5-small"
    tag = "best" if best else "latest"
    checkpoint_dir = os.path.join("checkpoints", "ft_experiments", args.experiment_name)
    model_path = os.path.join(checkpoint_dir, f"t5_{tag}.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {model_path}")

    print(f"🔄 Loading model checkpoint from {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)

    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    print(f"✅ Loaded {tag} model successfully on {DEVICE}")
    return model, tokenizer


# ===============================
#  Optimizer & Scheduler
# ===============================
def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler


def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad],
         "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    return optimizer


def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result

