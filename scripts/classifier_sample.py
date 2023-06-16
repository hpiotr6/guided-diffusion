"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data, get_data_loaders, join_from_tiles
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.gaussian_diffusion import num_back_steps
from PIL import Image

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    margin=10
    logger.log("loading data...")

    loaders = get_data_loaders(
        data_dir="datasets/GT-RAIN/GT-RAIN_test",
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=False,
        length=args.num_samples,
        margin=margin
    )

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []

    max_height = 0
    max_width = 0

    for i, (loader, width, height, grid) in enumerate(loaders):
        max_height = max(max_height, height)
        max_width = max(max_width, width)
        logger.log(f"sampling image no. {i+1}, width={width}, height={height}")
        data = list(loader)

        fragments = []
        for j, (batch, extra) in enumerate(data):
            if batch.shape[0] < args.batch_size:
                temp = th.zeros(args.batch_size, *batch.shape[1:])
                temp[:batch.shape[0], ...] = batch
                batch = temp

            batch = batch.to(dist_util.dev())

            model_kwargs = {}
            classes = th.randint(
                low=0, high=1, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            NUM_BACK_STEPS = num_back_steps(args.timestep_respacing)
            t = th.full((args.batch_size,), NUM_BACK_STEPS, device=dist_util.dev())
            batch = diffusion.q_sample(batch, t)
            shape = (args.batch_size, 3, args.image_size, args.image_size)
            # assert shape == batch.shape

            sample = sample_fn(
                model_fn,
                shape,
                noise=batch,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            for k, sample in enumerate(gathered_samples):
                for l, image in enumerate(sample.cpu().numpy()):
                    fragments.append(image)
                    out_path = os.path.join(logger.get_dir(), f"image_{i}_{j}_{k}_{l}.png")
                    Image.fromarray(image).save(out_path)
                    logger.log(f"created image_{i}_{j}_{k}_{l}.png")

            logger.log(f"number of fragments: {len(fragments)}")

        image = join_from_tiles(fragments, width, height, grid, margin).astype(np.uint8)
        out_path = os.path.join(logger.get_dir(), f"image_{i}.png")
        Image.fromarray(image).save(out_path)

        all_images.append(image)
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"joined {len(all_images)} images")

    arr = np.zeros((args.num_samples, max_height, max_width, 3))
    for i in range(args.num_samples):
        sh = all_images[i].shape
        arr[i,:sh[0],:sh[1],:] = all_images[i]/255

    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
