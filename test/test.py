import os
import time

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import disco


def aggregate(image_features, text_features):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # We gather tensors from all gpus to get more negatives to contrast with.
    gathered_image_features = [
        torch.zeros_like(image_features) for _ in range(world_size)
    ]
    gathered_text_features = [
        torch.zeros_like(text_features) for _ in range(world_size)
    ]
    dist.all_gather(gathered_image_features, image_features)
    dist.all_gather(gathered_text_features, text_features)

    all_image_features = torch.cat(
        [image_features] + 
        gathered_image_features[:rank]
        + gathered_image_features[rank + 1 :]
    )
    all_text_features = torch.cat(
        [text_features] + 
        gathered_text_features[:rank]
        + gathered_text_features[rank + 1 :]
    )
    return all_image_features, all_text_features


def aggregate_disco(image_features, text_features):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    bs = image_features.shape[0]

    # We gather tensors from all gpus to get more negatives to contrast with.
    gathered_image_features = [
        torch.zeros_like(image_features) for _ in range(world_size)
    ]
    gathered_text_features = [
        torch.zeros_like(text_features) for _ in range(world_size)
    ]
    dist.all_gather(gathered_image_features, image_features)
    dist.all_gather(gathered_text_features, text_features)

    gathered_image_features = torch.cat(gathered_image_features)
    gathered_text_features = torch.cat(gathered_text_features)
    
    all_image_features = gathered_image_features.requires_grad_(True)
    all_text_features = gathered_text_features.requires_grad_(True)

    image_features, text_features = all_image_features[bs*rank:bs*(rank+1)], all_text_features[bs*rank:bs*(rank+1)]

    return image_features, text_features, all_image_features, all_text_features


def main(gpu, ngpus_per_node, dist_url):
    dist.init_process_group(backend='nccl', init_method=dist_url,
                                    world_size=ngpus_per_node, rank=gpu)

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)
    mini_bs = 1024
    dim = 512

    criterion_img = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_text = torch.nn.CrossEntropyLoss(reduction='none')

    g = torch.Generator(device=device)
    g.manual_seed(10 + gpu)
    image_feature_ori = torch.randn(mini_bs, dim, requires_grad=True, generator=g, device=device)
    text_feature_ori = torch.randn(mini_bs, dim, requires_grad=True, generator=g, device=device)
    image_feature = image_feature_ori / image_feature_ori.norm(dim=-1, keepdim=True)
    text_feature = text_feature_ori / text_feature_ori.norm(dim=-1, keepdim=True)

    all_image_feature, all_text_feature = aggregate(image_feature, text_feature)

    logits_per_image = 100.0 * all_image_feature @ all_text_feature.t()
    logits_per_text = 100.0 * all_text_feature @ all_image_feature.t()
    # logits_per_text = logits_per_image.t()

    label = torch.arange(logits_per_image.shape[0]).long().to(device)

    loss1 = criterion_img(logits_per_image, label)
    loss2 = criterion_text(logits_per_text, label)

    loss = loss1 + loss2
    loss = loss.mean()
    loss.backward()

    time.sleep(gpu)
    print(gpu, image_feature_ori.grad.mean(), text_feature_ori.grad.mean())


def main_disco(gpu, ngpus_per_node, dist_url):
    dist.init_process_group(backend='nccl', init_method=dist_url,
                                    world_size=ngpus_per_node, rank=gpu)
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)
    bs = 1024
    dim = 512

    criterion_img = torch.nn.CrossEntropyLoss(reduction='mean')
    criterion_text = torch.nn.CrossEntropyLoss(reduction='mean')

    g = torch.Generator(device=device)
    g.manual_seed(10 + gpu)
    image_feature_ori = torch.randn(bs, dim, requires_grad=True, generator=g, device=device)
    text_feature_ori = torch.randn(bs, dim, requires_grad=True, generator=g, device=device)
    image_feature = image_feature_ori / image_feature_ori.norm(dim=-1, keepdim=True)
    text_feature = text_feature_ori / text_feature_ori.norm(dim=-1, keepdim=True)

    sub_image_features, sub_text_features, all_image_feature, all_text_feature = aggregate_disco(image_feature, text_feature)

    logits_per_image = 100 * sub_image_features @ all_text_feature.t()
    logits_per_text = 100 * sub_text_features @ all_image_feature.t()

    label = torch.arange(logits_per_image.shape[0]).long().to(device) + gpu * bs

    loss1 = criterion_img(logits_per_image, label)
    loss2 = criterion_text(logits_per_text, label)

    loss = loss1 + loss2
    loss.backward()

    image_grad = all_image_feature.grad
    text_grad = all_text_feature.grad

    torch.distributed.all_reduce(image_grad, op=torch.distributed.ReduceOp.AVG)
    torch.distributed.all_reduce(text_grad, op=torch.distributed.ReduceOp.AVG)

    image_feature.backward(image_grad[bs*gpu:bs*(gpu+1)])
    text_feature.backward(text_grad[bs*gpu:bs*(gpu+1)])

    time.sleep(gpu)
    print(gpu, image_feature_ori.grad.mean(), text_feature_ori.grad.mean())


def main_disco_gather(gpu, ngpus_per_node, dist_url):
    dist.init_process_group(backend='nccl', init_method=dist_url,
                                    world_size=ngpus_per_node, rank=gpu)

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)
    bs = 1024
    dim = 512
    rank = torch.distributed.get_rank()

    criterion_img = torch.nn.CrossEntropyLoss(reduction='mean')
    criterion_text = torch.nn.CrossEntropyLoss(reduction='mean')

    g = torch.Generator(device=device)
    g.manual_seed(10 + gpu)
    image_feature_ori = torch.randn(bs, dim, requires_grad=True, generator=g, device=device).float()
    text_feature_ori = torch.randn(bs, dim, requires_grad=True, generator=g, device=device).float()
    image_feature = image_feature_ori / image_feature_ori.norm(dim=-1, keepdim=True)
    text_feature = text_feature_ori / text_feature_ori.norm(dim=-1, keepdim=True)

    all_image_feature = disco.Gather(image_feature)
    all_text_feature = disco.Gather(text_feature)

    logits_per_image = 100.0 * all_image_feature[bs*rank:bs*(rank+1)] @ all_text_feature.t()
    logits_per_text = 100.0 * all_text_feature[bs*rank:bs*(rank+1)] @ all_image_feature.t()

    label = torch.arange(logits_per_image.shape[0]).long().to(device) + gpu * bs

    loss1 = criterion_img(logits_per_image, label)
    loss2 = criterion_text(logits_per_text, label)

    loss = loss1 + loss2
    loss.backward()

    time.sleep(gpu)
    print(gpu, image_feature_ori.grad.mean(), text_feature_ori.grad.mean())


def test():
    dist_url = "tcp://127.0.0.1:43261"
    ngpus_per_node = torch.cuda.device_count()

    print("origin clip:")
    mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, dist_url))
    torch.cuda.empty_cache()

    print("disco clip:")
    mp.spawn(main_disco, nprocs=ngpus_per_node, args=(ngpus_per_node, dist_url))
    torch.cuda.empty_cache()

    print("disco clip gather:")
    mp.spawn(main_disco_gather, nprocs=ngpus_per_node, args=(ngpus_per_node, dist_url))
    torch.cuda.empty_cache()

if __name__ == '__main__':
    test()

