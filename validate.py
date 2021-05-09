#!/usr/bin/env python
""" COCO validation script

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import torch
import torch.nn.parallel
from contextlib import suppress

from effdet import create_model, create_evaluator, create_dataset, create_loader
from effdet.data import resolve_input_config
from timm.utils import AverageMeter, setup_default_logging
from timm.models.layers import set_layer_config

from torchvision.ops.boxes import batched_nms
from effdet.config import get_efficientdet_config
from effdet.soft_nms import batched_soft_nms
from effdet.ensemble_boxes_wbf import *

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True


def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('root', metavar='DIR',
                    help='path to dataset root')
parser.add_argument('--dataset', default='coco', type=str, metavar='DATASET',
                    help='Name of dataset (default: "coco"')
parser.add_argument('--split', default='val',
                    help='validation split')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d1',
                    help='model architecture (default: tf_efficientdet_d1)')
add_bool_arg(parser, 'redundant-bias', default=None,
             help='override model config for redundant bias layers')
add_bool_arg(parser, 'soft-nms', default=None, help='override model config for soft-nms')
add_bool_arg(parser, 'wbf', default=None, help='override model config for wbf')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default='128', type=str,
                    metavar='N', help='mini-batch size (default: "128")')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results', default='./results.json', type=str, metavar='FILENAME',
                    help='JSON filename for evaluation results')


def validate(args):
    setup_default_logging()

    if args.amp:
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    args.pretrained = args.pretrained or not args.checkpoint  # might as well try to validate something
    args.prefetcher = not args.no_prefetcher

    dataset = create_dataset(args.dataset, args.root, args.split)

    evaluator = create_evaluator(args.dataset, dataset, pred_yxyx=False)

    flag_init = True
    output_dict = {}
    img_size_dict = {}

    image_size = list(get_efficientdet_config(args.model)['image_size'])
    assert isinstance(image_size, list), "image_size type must be tuple or list"
    print(f'multi image_size:{image_size}')

    batch_size = list(map(int, args.batch_size.split(',')))
    if len(batch_size) == 1:
        batch_size = batch_size * len(image_size)
    assert len(batch_size) == len(image_size), "batch_size should be a str splited by ','. like '2,4,8'"

    max_det_per_image = 1000

    for img_size, bs in zip(image_size, batch_size):
        print(f'img_size:{img_size} batch_size:{bs}')

        # create model
        with set_layer_config(scriptable=args.torchscript):
            # extra_args = {}
            # if args.img_size is not None:
            #     extra_args = dict(image_size=(args.img_size, args.img_size))
            bench = create_model(
                args.model,
                bench_task='predict',
                num_classes=args.num_classes,
                pretrained=args.pretrained,
                redundant_bias=args.redundant_bias,
                soft_nms=args.soft_nms,
                checkpoint_path=args.checkpoint,
                checkpoint_ema=args.use_ema,
                # **extra_args,
                image_size=img_size,
            )
        model_config = bench.config

        param_count = sum([m.numel() for m in bench.parameters()])
        print('Model %s created, param count: %d' % (args.model, param_count))

        bench = bench.cuda()

        amp_autocast = suppress
        if args.apex_amp:
            bench = amp.initialize(bench, opt_level='O1')
            print('Using NVIDIA APEX AMP. Validating in mixed precision.')
        elif args.native_amp:
            amp_autocast = torch.cuda.amp.autocast
            print('Using native Torch AMP. Validating in mixed precision.')
        else:
            print('AMP not enabled. Validating in float32.')

        if args.num_gpu > 1:
            bench = torch.nn.DataParallel(bench, device_ids=list(range(args.num_gpu)))

        input_config = resolve_input_config(args, model_config)
        loader = create_loader(
            dataset,
            input_size=input_config['input_size'],
            # batch_size=args.batch_size,
            batch_size=bs,
            use_prefetcher=args.prefetcher,
            interpolation=input_config['interpolation'],
            fill_color=input_config['fill_color'],
            mean=input_config['mean'],
            std=input_config['std'],
            num_workers=args.workers,
            pin_mem=args.pin_mem)

        bench.eval()
        batch_time = AverageMeter()
        end = time.time()

        last_idx = len(loader) - 1
        with torch.no_grad():
            for i, (input, target) in enumerate(loader):
                with amp_autocast():
                    output = bench(input, img_info=target)
                # evaluator.add_predictions(output, target)
                for idx, (img_idx, img_size) in enumerate(zip(target['img_idx'], target['img_size'])):
                    if flag_init:
                        output_dict[img_idx.item()] = [output[idx]]
                        img_size_dict[img_idx.item()] = img_size.cpu().numpy()
                    else:
                        output_dict[img_idx.item()].append(output[idx])
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.log_freq == 0 or i == last_idx:
                    print(
                        'Test: [{0:>4d}/{1}]  '
                        'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        .format(
                            i, len(loader), batch_time=batch_time,
                            rate_avg=input.size(0) / batch_time.avg)
                    )

        max_det_per_image = bench.max_det_per_image
        flag_init = False

    # nms or soft-nms for multi-scale
    print('Performing NMS:')
    for img_idx, _output in output_dict.items():
        output = torch.cat(_output, dim=0)

        if len(image_size) == 1:
            output = output.cpu().numpy()
            evaluator.img_indices.append(img_idx)
            evaluator.predictions.append(output)
            continue

        boxes, scores, classes = torch.split(output, [4, 1, 1], dim=1)
        scores = scores.reshape(-1)
        classes = classes.reshape(-1)

        if args.soft_nms:
            top_detection_idx, soft_scores = batched_soft_nms(
                boxes, scores, classes, method_gaussian=True, iou_threshold=0.3, score_threshold=.001)
            scores[top_detection_idx] = soft_scores
        else:
            top_detection_idx = batched_nms(boxes, scores, classes, iou_threshold=0.5)

        top_detection_idx = top_detection_idx[:max_det_per_image]
        boxes = boxes[top_detection_idx]
        scores = scores[top_detection_idx, None]
        classes = classes[top_detection_idx, None]  # +1 # back to class idx with background class = 0
        num_det = len(top_detection_idx)
        detections = torch.cat([boxes, scores, classes.float()], dim=1)
        if num_det < max_det_per_image:
            detections = torch.cat([
                detections,
                torch.zeros((max_det_per_image - num_det, 6), device=detections.device, dtype=detections.dtype)
            ], dim=0)
        detections = detections.cpu().numpy()

        evaluator.img_indices.append(img_idx)
        evaluator.predictions.append(detections)

    mean_ap = 0.
    if dataset.parser.has_labels:
        mean_ap = evaluator.evaluate()
    else:
        evaluator.save(args.results+'_nms.json')

    # return mean_ap

    if len(image_size) > 1 and args.wbf:
        mean_ap = 0.
        evaluator.reset()
        print('Performing WBF:')
        for img_idx, output in output_dict.items():
            if len(image_size) == 1:
                output = output.cpu().numpy()
                evaluator.img_indices.append(img_idx)
                evaluator.predictions.append(output)
                continue

            if img_idx % 100 == 0:
                print(f'processing {img_idx}')
            width, height = img_size_dict[img_idx]
            boxes, scores, classes = [], [], []
            for otpt in output:
                box, score, label = torch.split(otpt, [4, 1, 1], dim=1)
                box[:, 0] /= width
                box[:, 2] /= width
                box[:, 1] /= height
                box[:, 3] /= height
                boxes.append(box.cpu().numpy())
                scores.append(score.reshape(-1).cpu().numpy())
                classes.append(label.reshape(-1).cpu().numpy())
            boxes, scores, classes = weighted_boxes_fusion(boxes, scores, classes, weights=None, iou_thr=0.55,
                                                           skip_box_thr=0.01, conf_type='avg', allows_overflow=False)
            boxes[:, 0] *= width
            boxes[:, 2] *= width
            boxes[:, 1] *= height
            boxes[:, 3] *= height
            detections = np.concatenate((boxes, scores[:, None], classes[:, None]), axis=1)
            if detections.shape[0] == 0:
                continue
            evaluator.img_indices.append(img_idx)
            evaluator.predictions.append(detections)

        mean_ap = 0.
        if dataset.parser.has_labels:
            mean_ap = evaluator.evaluate()
        else:
            evaluator.save(args.results+'_wbf.json')

    return mean_ap


def main():
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()
