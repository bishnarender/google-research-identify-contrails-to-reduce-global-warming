## google-research-identify-contrails-to-reduce-global-warming
## score at 4th position is achieved.
![icrgw-submission](https://github.com/bishnarender/google-research-identify-contrails-to-reduce-global-warming/assets/49610834/1d222a45-d081-4a5d-bab5-8e29ae3fb2bb)

### Start 
-----
For better understanding of project, read the files in the following order:
1. eda.ipynb 
2. all_in_one.ipynb
3. google-icrgw-submission.ipynb

Only band_11, band_14 and band_15 are selected for the model as these bands have "highest value", "more value fluctuations" and "high positive correlation with contrails".

Further some information about bands is as follows:
1. Band 11 is often used to discern the physical state of cloud tops, whether they are in a liquid or frozen phase. This is essential for understanding cloud structure and storm intensity.
2. Band 14 observes in the shortwave infrared spectrum, allowing for the detection of hot spots such as fires, volcanic activity, and land temperature.
3. Sometimes known as the "dirty" longwave band, Band 15 can detect certain atmospheric particles and moisture content, making it useful for distinguishing between different cloud types and studying the atmosphere's overall moisture profile.
Ref: https://www.goes-r.gov/spacesegment/ABI-tech-summary.html

4 time-steps are chosen out of 8 i.e., chosen indices 1, 2, 3 and 4. Masks provided by the organizer were created using the image present at 5th time-step i.e., at 4th index.
<code>
a = np.load('%s/band_%02d.npy' % (path, k))[:, :, 1:5]
</code>

### ViT 
-----
![ic-](https://github.com/bishnarender/google-research-identify-contrails-to-reduce-global-warming/assets/49610834/28482b87-6d55-4c2d-8101-78054602783f)

Original input is in the form [4, 3, 256, 256] where 4 denotes 4 time-steps and 3 denotes 3 bands (i.e., band_11, band_14 and band_15). It is resized to [4, 3, 512, 512] using torchvision.transforms. This is further transformed to model input [3, 1024, 1024] as follows:
<code>
x4 = np.zeros((3, 1024, 1024), dtype=np.float32)
#- nc = 512
x4[:, :nc, :nc] = x[3]  #- t=4  #- top-left section of 1024 x 1024 
x4[:, :nc, nc:] = x[0]  #- top-right section of 1024 x 1024 
x4[:, nc:, :nc] = x[1]  #- bottom-left section of 1024 x 1024 
x4[:, nc:, nc:] = x[2]  #- bottom-right section of 1024 x 1024
#- x4.shape => (3, 1024, 1024)
</code>

nn.Identity() is a “pass-through” layer will just return the input without any manipulation and can be used to e.g. replace other layers.

y is mean of 4 annotations provided in file human_individual_masks.npy. y_sym has size 512×512 and is sampled from y on a "shifted (0.5) regular grid" with bilinear interpolation.
<code>
y_sym = F.grid_sample(y.unsqueeze(0), self.grid, mode=self.y_sym_mode, padding_mode='border', align_corners=False).squeeze(0)
</code>
As, the rotated label is shifted up compared to the to their expected position when augmentation is applied to the input image. The 0.5 shift is necessary because without this score dropped with flip and rot90 (multiples of 90° rotation) augmentations. This phenomenon happens because of the uneven structure of contrails. The phenomenon is confirmed by applying augmentation to a 180°-rotated input image to see why the augmentation did not work.

Final backward loss is computed as:
<code>
criterion = nn.BCEWithLogitsLoss(reduction='none')
loss_sym = criterion(y_sym_pred, y_sym).mean(dim=(1, 2, 3))
loss_original = criterion(y_pred, y).mean(dim=(1, 2, 3))
loss = loss_sym + w * loss_original #- w is 0 when augmentation is applied.
#- loss.shape => torch.Size([BS])
loss = loss.mean()  
</code>

#### MaxViT: Multi-Axis Vision Transformer
-----
MaxViT is a family of hybrid (CNN + ViT) image classification model.

ViT struggle to scale to higher input resolution because the computational complexity increases quadratically with respect to the image size. MaxViT can better adapt to high-resolution, dense prediction tasks, and can naturally adapt to different input sizes with high flexibility and low complexity. It significantly improves the state of the art for high-level tasks, such as image classification, object detection, segmentation, quality assessment, and generation.

The biggest problem with using Vision Transformers for image is that Global Self attention have "quadratic complexity" and using localized attention in patches makes the network lose the global context of the image. It is not possible to apply "quadratic complex" self attention in in the initial layers of the network and possible only in the deeper layers of the network. This limits the ability to use Visual Transformers as the backbone network for vision tasks.

Here the new approach is based on multi-axis attention (Max-SA), which decomposes the full-size attention (each pixel attends to all the pixels) used in ViT into two sparse forms — local and (sparse/dilated) global. This reduces the quadratic complexity of vanilla/vit attention to linear, without any loss of non-locality. As shown in the figure below, the multi-axis attention contains a sequential stack of "block attention" and "grid attention". The block attention works within non-overlapping windows (small patches in intermediate feature maps) to capture local patterns, while the grid attention works on a <b>sparsely sampled</b> uniform grid for long-range (global) interactions. The window sizes (P and G) of "grid and block attentions" can be fully controlled as hyperparameters to ensure a linear computational complexity to the input size.

The same colors are spatially mixed by the self-attention operation.
![max](https://github.com/bishnarender/google-research-identify-contrails-to-reduce-global-warming/assets/49610834/8847ee7e-70bd-4c02-9630-ca01d8c771ee)
[Image Reference](https://blog.research.google/2022/09/a-multi-axis-approach-for-vision.html)

In MaxViT, author first build a single MaxViT block (shown below) by concatenating MBConv (proposed by EfficientNet, V2) with the multi-axis attention. This single block can encode local and global visual information regardless of input resolution. Author then simply stack repeated blocks composed of attention and convolutions in a hierarchical architecture (similar to ResNet, CoAtNet), yielding our homogenous MaxViT architecture. Notably, MaxViT is distinguished from previous hierarchical approaches as it can “see” globally throughout the entire network, even in earlier, high-resolution stages, demonstrating stronger model capacity on various tasks.

![maxvit](https://github.com/bishnarender/google-research-identify-contrails-to-reduce-global-warming/assets/49610834/a6a50b7a-d9a0-4bf5-b202-ae4d95a2f1c2)
[Image Reference](https://blog.research.google/2022/09/a-multi-axis-approach-for-vision.html)

Let X be input feature map of shape [224,224,3] i.e., HxWxC. Instead of applying self-attention on the flattened spatial dimension 50176 (HW), author block the feature into a tensor of shape [(224/7)×(224/7),7×7,3] i.e., [1024, 49, 3] where P=7. ​Note that after window/block partition, the block dimensions are gathered onto the spatial dimension (i.e., -2 axis). This represents partitioning into non-overlapping windows​/blocks, each of size 7×7. Applying self-attention on the local spatial dimension i.e., 7 × 7, is equivalent to attending within a small window. We will use this <b>block attention</b> to conduct local interactions. 

Inspired by block attention, we present a surprisingly simple but effective way to gain sparse global attention, which we call grid attention. Instead of partitioning the feature map using a fixed window (comprising only the neigbours), we partitioned using a full window with samples collected sparsely. Thus, we grid feature into a tensor of shape [7×7, (224/7)×(224/7), 3] i.e., [49, 1024, 3] where G=7. Transpose to place the grid dimension in the assumed spatial axis (i.e., -2 axis) as [1024, 49, 3]. Using a fixed 7×7 uniform grid, resulting in windows/blocks having adaptive size 49×49.  Employing self-attention on the decomposed grid axis i.e., 7×7, corresponds to dilated, global spatial mixing of tokens. 

By using the same fixed window and grid sizes (we use P = G = 7 following Swin), we can fully balance the computation between local and global operations, both having only linear complexity with respect to spatial size or sequence length. Yet it enjoys global interaction capability without requiring masking, padding, or cyclic-shifting, making it more implementation friendly, preferable to the shifted window scheme.

Using MBConv layers prior to attention offers another advantage, in that depthwise convolutions can be regarded as conditional position encoding (CPE), making our model free of explicit positional encoding layers.

The relative attention can be defined as:
<p align="center">RelAttention(Q, K, V ) = softmax(QK<sup>T</sup>/√d + B)V,</p>
where Q, K, V ε R<sup>(H×W)×C</sup> are the query, key, and value matrices and d is the hidden dimension. The attention weights are co-decided by a learned static location-aware matrix B and the scaled input-adaptive attention QK<sup>T</sup>/√d. We assume the relative attention operator above follows the convention for 1D input sequences i.e., always regards the second last dimension of an input (..., L, C) as the spatial axis where L, C represent sequence length and channels. The proposed Multi-Axis Attention can be implemented without modification to this self-attention operation. 

Multi-axis attention module. Given an input tensor x ε R<sup>H×W×C</sup>, the local Block Attention can be expressed as:
<p align="center">x ← x + Unblock(RelAttention(Block(LN(x))))</p>
<p align="center">x ← x + MLP(LN(x)) </p>
while the global, sparse/dilated Grid Attention module is formulated as:
<p align="center">x ← x + Ungrid(RelAttention(Grid(LN(x))))</p>
<p align="center">x ← x + MLP(LN(x)) </p>
<p>where we omit the QKV input format in the RelAttention operation for simplicity. A RelAttention operation applied on the -2 axis. LN denotes the Layer Normalization, where MLP is a standard MLP network consisting of two linear layers: x ← W2GELU(W1x). Block converts input [H,W,C] to shape [(H/7)*(W/7),7*7,3]. Unblock converts input [(H/7)*(W/7),7*7,3] to shape [H,W,C]. The same is for grid and ungrid.</p>

Instead of firstly flattening the multi-dimensional input/map as ViT does, Axial attention of some models applies self-attention first column-wise at a time and then row-wise, finally combining the attention maps of multiple axes to achieve a global receptive field.

But here, Multi-axis attention first employs "block attention" (with receptive field 7x7 having center as our target pixel) to conduct local interactions. Then Multi-axis attention employs "sparse/dilated global attention" (with receptive field 7x7 constituting distance pixels and having center as our target pixel) to conduct global interactions. Thus enjoying global receptive fields in two steps. Receptive field w.r.t attention means pixels at which we are paying attention for our target pixel.

![max_final](https://github.com/bishnarender/google-research-identify-contrails-to-reduce-global-warming/assets/49610834/49235478-dba3-4fff-a0ae-737b958efd0e)

​In the above image, the area marked pink is our target pixel. Both "block attention" and "grid attention" receptive fields are shown in a single image surrounding the target pixel.
