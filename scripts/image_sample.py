"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os

from PIL import Image
from argparse import ArgumentParser
import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

cwd = os.getcwd()

def set_seed(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def save_image_from_array(array, path):
    image = Image.fromarray(array)
    image.save(path)


def main():
    args = create_argparser().parse_args()
    os.mkdir(f"{cwd}/samples/{args.name}")
    if args.seed is not None:
        set_seed(args.seed)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    image_count = 0
    max_images = 100
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)  # Adjusting the tensor for PIL compatibility
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL

        # Inside the while loop, right after generating samples
        for sample_batch in gathered_samples:
            for sample in sample_batch:
                image_count += 1
                if image_count > max_images:
                    break
                image_np = sample.cpu().numpy()
                # Ensure the image is in (height, width, channels) format
                if image_np.shape[2] == 1:  # If grayscale, convert to RGB for compatibility
                    image_np = np.repeat(image_np, 3, axis=2)

                if image_np.shape[0] > 1 and image_np.shape[1] > 1:  # Basic check to ensure image shape makes sense
                    image_path = f"{cwd}/samples/{args.name}/sample_{image_count:04d}.png"
                    save_image_from_array(image_np, image_path)
                else:
                    logger.log("Invalid image shape encountered", image_np.shape)

        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        if image_count >= max_images:
            break
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        save_dir = os.path.join(cwd, f"samples/{args.name}")
        out_path = os.path.join(save_dir, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=16,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='Seed for random number generators for reproducibility')
    parser.add_argument('--name', type=str, help='Seed for random number generators for reproducibility')
    
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
