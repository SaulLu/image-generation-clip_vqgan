#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Script to generate an image from a text prompt

Code inspired initially by a notebook made by Katherine Crowson 
"""
import argparse
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

import dm_pix as pix
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import core, struct
from flax.training.common_utils import get_metrics
from jax import custom_vjp
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CLIPFeatureExtractor,
    CLIPProcessor,
    CLIPTokenizer,
    CLIPTokenizerFast,
    FlaxCLIPModel,
    HfArgumentParser,
    is_tensorboard_available,
    set_seed,
)
from vqgan_jax.modeling_flax_vqgan import VQModel


@dataclass
class ModelArguments:
    """
    Arguments
    """

    clip_model_name_or_path: Optional[str] = field(
        default="openai/clip-vit-base-patch32",
        metadata={"help": "The model checkpoint for weights initialization of CLIP model."},
    )
    clip_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as clip_model_name_or_path"}
    )
    clip_tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path used to tokenize the textual prompts if not the same as clip_model_name_or_path"
        },
    )
    vqgan_model_name_or_path: Optional[str] = field(
        default="valhalla/vqgan-imagenet-f16-1024",
        metadata={"help": "The model checkpoint for weights initialization of VQGAN model."},
    )
    vqgan_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as vqgan_model_name_or_path"}
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=False,  # todo: change
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class ImageGenerationArguments:
    texts_prompts: List[str] = field(
        metadata={"help": "the list of texts that the generated image should look like."},
    )
    image_width: Optional[int] = field(
        default=480,
        metadata={"help": "The width of the generated image."},
    )
    image_height: Optional[int] = field(
        default=480,
        metadata={"help": "The height of the generated image."},
    )


@dataclass
class TrainingArguments:

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    learning_rate: float = field(default=0.05, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})

    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    # lr_scheduler_type: SchedulerType = field(
    #     default="linear",
    #     metadata={"help": "The scheduler type to use."},
    # )
    # warmup_ratio: float = field(
    #     default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    # )
    # warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    logging_steps: int = field(default=1, metadata={"help": "Log every X updates steps."})
    logging_steps_heavy: int = field(default=10, metadata={"help": "Log every X updates steps heavy data such as image."})
    save_steps: int = field(default=0, metadata={"help": "Save image every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of images."
                "Deletes the older images in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    cut_num: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "The number of random samples to extract from the image decoded by VQGAN in order "
                "to compute their embeddings with CLIP and compare each of these embeddings with "
                "each of the textual prompts embeddings."
            )
        },
    )
    n_crop_sizes:  Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "The number of possible size (jax constrain) #TODO polish"
            )
        },
    )


class TrainState(struct.PyTreeNode):
    """Simple train state for the common case with a single Optax optimizer.

    Synopsis::

        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=tx)
        grad_fn = jax.grad(make_loss_fn(state.apply_fn))
        for batch in data:
            grads = grad_fn(state.params, batch)
            state = state.apply_gradients(grads=grads)

    Note that you can easily extend this dataclass by subclassing it for storing
    additional data (e.g. additional variable collections).

    For more exotic usecases (e.g. multiple optimizers) it's probably best to
    fork the class and modify it.

    Args:
        step: Counter starts at 0 and is incremented by every call to
        `.apply_gradients()`.
        apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
        convenience to have a shorter params list for the `train_step()` function
        in your training loop.
        params: The parameters to be updated by `tx` and used by `apply_fn`.
        tx: An Optax gradient transformation.
        opt_state: The state for `tx`.
    """

    step: int
    params: core.FrozenDict[str, Any]
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
        grads: Gradients that have the same pytree structure as `.params`.
        **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
        An updated instance of `self` with `step` incremented by one, `params`
        and `opt_state` updated by applying `grads`, and additional attributes
        replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


# f :: a -> b
@custom_vjp
def clip_with_grad(x):
    return jnp.clip(x, a_min=0, a_max=1)


# f_fwd :: a -> (b, c)
def clip_with_grad_fwd(x):
    return clip_with_grad(x), x


# f_bwd :: (c, CT b) -> CT a
def clip_with_grad_bwd(x, y_bar):
    ans = clip_with_grad(x)
    boolean = jnp.heaviside(y_bar * (x - ans), 1)
    ans_dot = y_bar * boolean
    return (ans_dot,)


clip_with_grad.defvjp(clip_with_grad_fwd, clip_with_grad_bwd)


def resample(input, size, align_corners=True):
    return jax.image.resize(input, size, method="bicubic")


def resized_and_crop(img, rng, final_shape, crop_sizes):
    # size = jax.random.randint(subrng, shape=(1,), minval=min_size, maxval=max_size).item()
    rng, subrng = jax.random.split(rng)
    cutout = pix.random_crop(key=subrng, image=img, crop_sizes=crop_sizes)

    # offsetx = jax.random.randint(subrng, shape=(1,), minval=0, maxval=sideX - size + 1).item()

    # rng, subrng = jax.random.split(rng)
    # offsety = jax.random.randint(subrng, shape=(1,), minval=0, maxval=sideY - size + 1).item()
    # cutout = img[:, :, offsetx : offsetx + size, offsety : offsety + size]

    # resize
    return resample(cutout, final_shape)


def get_crop_sizes(image_width, image_height, min_image_width_height, n_crop_sizes):
    max_size = min(image_width, image_height)
    min_size = min(image_width, image_height, min_image_width_height)

    all_possibilities = list(range(min_size, max_size + 1))
    if len(all_possibilities) < n_crop_sizes:
        raise ValueError(f"`n_crop_sizes` {n_crop_sizes} must be superior or equal to {len(all_possibilities)}")
    crop_sizes = random.sample(all_possibilities, n_crop_sizes)
    crop_sizes = [(1,3, crop_size, crop_size) for crop_size in crop_sizes]
    return crop_sizes


def random_resized_crop(img, rng, image_width_height_clip, n_subimg, crop_sizes):
    # sideY, sideX = img.shape[2:4]
    # max_size = min(sideX, sideY)
    # min_size = min(sideX, sideY, shape[0])

    # rng, subrng = jax.random.split(rng)
    # # size = jax.random.choice(subrng, jnp.array([min_size, max_size])).item()
    # size = jax.random.randint(subrng, shape=(1,), minval=min_size, maxval=max_size).item()

    final_shape = img.shape
    final_shape = jax.ops.index_update(final_shape, jax.ops.index[-2], image_width_height_clip)
    final_shape = jax.ops.index_update(final_shape, jax.ops.index[-1], image_width_height_clip)

    metrics = {}
    cutouts = []

    for i in range(n_subimg):
        rng, subrng = jax.random.split(rng)
        cutout = resized_and_crop(img, subrng, final_shape, crop_sizes=crop_sizes)
        cutouts.append(cutout)

    cutouts = jnp.concatenate(cutouts, axis=0)
    return cutouts, metrics


def speric_distance(embed_img):
    dist = jnp.add(embed_img, -text_embeds)
    dist = jax.numpy.linalg.norm(dist, ord=2, axis=-1)
    return jnp.arcsin(dist / 2) ** 2 * 2


if __name__ == "__main__":
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, ImageGenerationArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level="NOTSET",
        datefmt="[%X]",
    )

    # Log on each process the small summary:
    logger = logging.getLogger(__name__)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    logger.info(f"Training/evaluation parameters {training_args}")

    # Load clip tokenizer
    if model_args.clip_tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            model_args.clip_tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.clip_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            model_args.clip_model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "This script does not train a model, you must choose already trained tokenizers, using --tokenizer_name."
        )

    # Load CLIP model
    if model_args.clip_config_name:
        clip_config = AutoConfig.from_pretrained(model_args.clip_config_name, cache_dir=model_args.cache_dir)
    elif model_args.clip_model_name_or_path:
        clip_config = AutoConfig.from_pretrained(model_args.clip_model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "This script does not train a model, you must choose an already trained CLIP model, using --clip_model_name_or_path."
        )

    if model_args.clip_model_name_or_path:
        clip_model = FlaxCLIPModel.from_pretrained(
            model_args.clip_model_name_or_path,
            config=clip_config,
        )
    else:
        raise ValueError(
            "This script does not train a model, you must choose an already trained CLIP model, using --clip_model_name_or_path."
        )

    # Uncomment when VQ-GAN merged into transformers
    # Load VQGAN model
    # if model_args.vqgan_config_name:
    #     vqgan_config = AutoConfig.from_pretrained(model_args.vqgan_config_name, cache_dir=model_args.cache_dir)
    # elif model_args.vqgan_model_name_or_path:
    #     vqgan_config = AutoConfig.from_pretrained(model_args.vqgan_model_name_or_path, cache_dir=model_args.cache_dir)
    # else:
    #     raise ValueError(
    #         "This script does not train a model, you must choose an already trained CLIP model, using --vqgan_model_name_or_path."
    #     )

    if model_args.vqgan_model_name_or_path:
        vqgan_model = VQModel.from_pretrained(
            model_args.vqgan_model_name_or_path,
            # config=vqgan_config, # not merged yet
        )
    else:
        raise ValueError(
            "This script does not train a model, you must choose an already trained VQGAN model, using --vqgan_model_name_or_path."
        )

    # init logging utils
    combined_dict = {
        **asdict(model_args),
        **asdict(data_args),
        **asdict(training_args),
    }
    wandb.init(project="vqgan-clip", dir=training_args.output_dir)
    wandb.config.update(combined_dict, allow_val_change=True)

    context_length = clip_model.config.text_config.max_position_embeddings

    cut_size = clip_model.config.vision_config.image_size
    e_dim = vqgan_model.config.embed_dim

    f = 2 ** (vqgan_model.config.num_resolutions - 1)

    n_toks = vqgan_model.config.n_embed
    toksX, toksY = data_args.image_width // f, data_args.image_height // f
    sideX, sideY = toksX * f, toksY * f

    # not used
    # z_min = jnp.min(model.params["quantize"]["embedding"]["embedding"], axis=0)
    # z_max = jnp.max(model.params["quantize"]["embedding"]["embedding"], axis=0)

    # Create the text CLIP embeddings of the textual prompts
    def create_text_embedding(batch):
        text_embeds = clip_model.get_text_features(**batch)

        # normalized features
        text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)

        return text_embeds

    inputs = tokenizer(data_args.texts_prompts, padding="max_length", max_length=context_length, return_tensors="jax")
    text_embeds = create_text_embedding(inputs)

    rng = jax.random.PRNGKey(training_args.seed)

    # Create a first random VQGAN latent image representation
    rng, subrng = jax.random.split(rng)
    one_hot = jax.nn.one_hot(jax.random.randint(subrng, [toksY * toksX], 0, n_toks), n_toks)
    z = jnp.matmul(one_hot, vqgan_model.params["quantize"]["embedding"]["embedding"])
    z = jnp.reshape(z, (-1, toksY, toksX, e_dim))

    # Initialize optimizer
    optimizer = optax.adamw(
        learning_rate=training_args.learning_rate,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    state = TrainState.create(params=z, tx=optimizer)

    def straight_through_quantize(x):
        return x + jax.lax.stop_gradient(vqgan_model.quantize(x)[0] - x)

    clip_get_image_features_fn = jax.jit(clip_model.get_image_features)
    vqgan_decode_fn = jax.jit(vqgan_model.decode)
    vqgan_quantize_fn = jax.jit(straight_through_quantize)

    possible_crop_sizes = get_crop_sizes(data_args.image_width, data_args.image_height, cut_size, n_crop_sizes=training_args.n_crop_sizes)

    def train_step(rng, state, text_embeds, n_subimg, crop_sizes):
        def loss_fn(params, rng):
            z_latent_q = vqgan_quantize_fn(params)
            output_vqgan_decoder = clip_with_grad((vqgan_decode_fn(z_latent_q) + 1) / 2)  # deterministic ??

            output_vqgan_decoder_reshaped = jnp.moveaxis(output_vqgan_decoder, (2, 1), (3, 2))

            rng, subrng = jax.random.split(rng)
            crop_sizes = jax.random.choice(possible_crop_sizes)
            imgs_stacked, metrics = random_resized_crop(
                output_vqgan_decoder_reshaped,
                subrng,
                image_width_height_clip=cut_size,
                n_subimg=n_subimg,
                crop_sizes=crop_sizes,
            )
            image_embeds = clip_get_image_features_fn(pixel_values=imgs_stacked)

            # normalized features
            image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)

            # compute distance
            dists = jax.vmap(speric_distance, in_axes=0)(image_embeds)

            loss = dists.mean()
            return loss, (output_vqgan_decoder, metrics)

        rng, subrng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (output_vqgan_decoder, metrics)), grad = grad_fn(state.params, subrng)

        new_state = state.apply_gradients(grads=grad)

        metrics.update({"loss": loss, "step": state.step, "image": output_vqgan_decoder})

        return new_state, metrics

    compt = 0
    stop_training = False
    try:
        train_time = 0
        while not stop_training:
            compt += 1
            # ======================== Training ================================
            train_start = time.time()

            rng, subrng = jax.random.split(rng)
            state, train_metric = train_step(
                rng=subrng,
                state=state,
                text_embeds=text_embeds,
                n_subimg=training_args.cut_num,
                crop_sizes=crop_sizes
            )

            train_time_step = time.time() - train_start
            train_time += train_time_step

            # trick, not used
            # state.replace(params= jnp.clip(state.params, a_min=z_min, a_max=z_max))

            # Save metrics
            if jax.process_index() == 0 and compt % training_args.logging_steps == 0:
                train_metric.update({"time": train_time, "train_time_step": train_time_step})
                if compt%training_args.logging_steps_heavy:
                    train_metric["image"] = wandb.Image(
                        Image.fromarray(np.asarray((train_metric["image"][0] * 255).astype(np.uint8)))
                    )
                train_metric["loss"] = np.asarray(train_metric["loss"])
                wandb.log(train_metric)
            if training_args.max_steps > 0 and compt > training_args.max_steps:
                stop_training = True
    except KeyboardInterrupt:
        pass
