# DisCo-CLIP

Official PyTorch implementation of the paper "DisCo-CLIP: A Distributed Contrastive Loss for Memory Efficient CLIP Training".

![DisCo-CLIP](https://github.com/IDEA-Research/DisCo-CLIP/blob/chenyihao/DisCo-CLIP.png)

### Installation
```
git clone https://github.com/IDEA-Research/DisCo-CLIP.git
cd DisCo-CLIP
pip install -e .
```


### Usage
We implemented our method using `disco.Gather` , which is easy to use. more detail about `disco.Gather`  in [gather.py]([https://www.a.com](https://github.com/IDEA-Research/DisCo-CLIP/blob/main/disco/gather.py))

```python

import disco

...

all_image_feature = disco.Gather(image_feature)
all_text_feature = disco.Gather(text_feature)

# bs is batch size per gpu
# rank is global rank
logits_per_image = 100.0 * all_image_feature[bs*rank:bs*(rank+1)] @ all_text_feature.t()
logits_per_text = 100.0 * all_text_feature[bs*rank:bs*(rank+1)] @ all_image_feature.t()

label = torch.arange(logits_per_image.shape[0]).long().to(device) + rank * bs

loss1 = criterion_img(logits_per_image, label)
loss2 = criterion_text(logits_per_text, label)

loss = loss1 + loss2
loss.backward()


```


### Citation
If you find this repository helpful, please consider citing:

```
@Article{chen2023discoclip,
  author  = {Yihao Chen and Xianbiao Qi and Jianan Wang and Lei Zhang},
  title   = {DisCo-CLIP: A Distributed Contrastive Loss for Memory Efficient CLIP Training},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2023},
}

```
