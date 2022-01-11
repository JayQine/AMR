import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import voc12.dataloader
from misc import torchutils, imutils
import cv2
cudnn.enabled = True
from step.gradCAM import GradCAM


def adv_climb(image, epsilon, data_grad):
    sign_data_grad = data_grad / (torch.max(torch.abs(data_grad))+1e-12)
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, image.min().data.cpu().float(), image.max().data.cpu().float()) # min, max from data normalization
    return perturbed_image

def add_discriminative(expanded_mask, regions, score_th):
    region_ = regions / regions.max()
    expanded_mask[region_>score_th]=1
    return expanded_mask

def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=True)
    
    with cuda.device(process_id):
        model.cuda()
        gcam = GradCAM(model=model, candidate_layers=["stage4", "stage2_4"])
        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            if os.path.exists(os.path.join(args.cam_out_dir, img_name + '.npy')):
                continue
            size = pack['size']
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            outputs_cam1 = []
            outputs_cam2 = []
            n_classes = len(list(torch.nonzero(pack['label'][0])[:, 0]))

            # 1
            for s_count, size_idx in enumerate([1, 2, 0, 3, 4, 5, 6]):
                orig_img = pack['img'][size_idx].clone()
                for c_idx, c in enumerate(list(torch.nonzero(pack['label'][0])[:, 0])):
                    pack['img'][size_idx] = orig_img
                    img_single = pack['img'][size_idx].detach()[0]  

                    if size_idx != 1:
                        total_adv_iter = args.adv_iter
                    else:
                        if args.adv_iter > 10:
                            total_adv_iter = args.adv_iter // 2
                            mul_for_scale = 2
                        elif args.adv_iter < 6:
                            total_adv_iter = args.adv_iter
                            mul_for_scale = 1
                        else:
                            total_adv_iter = 5
                            mul_for_scale = float(total_adv_iter) / 5

                    for it in range(total_adv_iter):
                        img_single.requires_grad = True

                        outputs = gcam.forward(img_single.cuda(non_blocking=True), step=1)

                        if c_idx == 0 and it == 0:
                            cam_all_classes = torch.zeros([n_classes, outputs.shape[2], outputs.shape[3]])

                        gcam.backward(ids=c)

                        regions = gcam.generate(target_layer="stage4")
                        regions = regions[0] + regions[1].flip(-1)

                        if it == 0:
                            init_cam = regions.detach()

                        cam_all_classes[c_idx] += regions[0].data.cpu() * mul_for_scale
                        logit = outputs
                        logit = F.relu(logit)
                        logit = torchutils.gap2d(logit, keepdims=True)[:, :, 0, 0]

                        valid_cat = torch.nonzero(pack['label'][0])[:, 0]
                        logit_loss = - 2 * (logit[:, c]).sum() + torch.sum(logit)

                        expanded_mask = torch.zeros(regions.shape)
                        expanded_mask = add_discriminative(expanded_mask, regions, score_th=args.score_th)

                        L_AD = torch.sum((torch.abs(regions - init_cam))*expanded_mask.cuda())

                        loss = - logit_loss - L_AD * args.AD_coeff

                        model.zero_grad()
                        img_single.grad.zero_()
                        loss.backward()

                        data_grad = img_single.grad.data

                        perturbed_data = adv_climb(img_single, args.AD_stepsize, data_grad)
                        img_single = perturbed_data.detach()

                outputs_cam1.append(cam_all_classes)

            strided_cam1 = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs_cam1]), 0)
            highres_cam1 = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs_cam1]

            highres_cam1 = torch.sum(torch.stack(highres_cam1, 0), 0)[:, 0, :size[0], :size[1]]
            strided_cam1 /= F.adaptive_max_pool2d(strided_cam1, (1, 1)) + 1e-5
            highres_cam1 /= F.adaptive_max_pool2d(highres_cam1, (1, 1)) + 1e-5

            # 2
            for s_count, size_idx in enumerate([1, 2, 0, 3, 4, 5, 6]):
                orig_img = pack['img'][size_idx].clone()
                for c_idx, c in enumerate(list(torch.nonzero(pack['label'][0])[:, 0])):
                    pack['img'][size_idx] = orig_img
                    img_single = pack['img'][size_idx].detach()[0]  

                    if size_idx != 1:
                        total_adv_iter = args.adv_iter
                    else:
                        if args.adv_iter > 10:
                            total_adv_iter = args.adv_iter // 2
                            mul_for_scale = 2
                        elif args.adv_iter < 6:
                            total_adv_iter = args.adv_iter
                            mul_for_scale = 1
                        else:
                            total_adv_iter = 5
                            mul_for_scale = float(total_adv_iter) / 5

                    for it in range(total_adv_iter):
                        img_single.requires_grad = True

                        outputs = gcam.forward(img_single.cuda(non_blocking=True), step=2)

                        if c_idx == 0 and it == 0:
                            cam_all_classes = torch.zeros([n_classes, outputs.shape[2], outputs.shape[3]])

                        gcam.backward(ids=c)

                        regions = gcam.generate(target_layer="stage2_4")
                        regions = regions[0] + regions[1].flip(-1)

                        if it == 0:
                            init_cam = regions.detach()

                        cam_all_classes[c_idx] += regions[0].data.cpu() * mul_for_scale
                        logit = outputs
                        logit = F.relu(logit)
                        logit = torchutils.gap2d(logit, keepdims=True)[:, :, 0, 0]

                        valid_cat = torch.nonzero(pack['label'][0])[:, 0]
                        logit_loss = - 2 * (logit[:, c]).sum() + torch.sum(logit)

                        expanded_mask = torch.zeros(regions.shape)
                        expanded_mask = add_discriminative(expanded_mask, regions, score_th=args.score_th)

                        L_AD = torch.sum((torch.abs(regions - init_cam))*expanded_mask.cuda())

                        loss = - logit_loss - L_AD * args.AD_coeff

                        model.zero_grad()
                        img_single.grad.zero_()
                        loss.backward()

                        data_grad = img_single.grad.data

                        perturbed_data = adv_climb(img_single, args.AD_stepsize, data_grad)
                        img_single = perturbed_data.detach()

                outputs_cam2.append(cam_all_classes)

            strided_cam2 = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs_cam2]), 0)
            highres_cam2 = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs_cam2]

            highres_cam2 = torch.sum(torch.stack(highres_cam2, 0), 0)[:, 0, :size[0], :size[1]]
            strided_cam2 /= F.adaptive_max_pool2d(strided_cam2, (1, 1)) + 1e-5
            highres_cam2 /= F.adaptive_max_pool2d(highres_cam2, (1, 1)) + 1e-5

            strided_cam_weight = strided_cam1 * args.weight + strided_cam2 * (1-args.weight)
            highres_cam_weight = highres_cam1 * args.weight + highres_cam2 * (1-args.weight)
            
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam_weight.cpu(), "high_res": highres_cam_weight.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.amr_network), 'CAM')()
    model.load_state_dict(torch.load(args.amr_weights_name + '.pth'), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()