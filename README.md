![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/28a5ecf68b8d4b9da0687f9e5e555c76~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgdWhha2Fkb3Rjb20=:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjg3NTk3ODE0NzY5MjkxMCJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1730205320&x-orig-sign=fZcgDNv1I2U5vpW0X0aMJIXSJwI%3D)

# CLIP算法简介

OpenAI的CLIP模型是一种强大的多模态深度学习工具，能够有效地处理图像和文本之间的关系。通过在4亿对图文数据上进行训练，CLIP实现了图像与文本的高效匹配，展现出广泛的应用前景，包括零次学习图像分类、文本到图像检索、视觉问答等多个领域。

## 视觉问题回答与描述生成

在视觉问题回答领域，CLIP模型通过将图像和问题文本编码到同一空间中，能够找到与问题最相关的图像区域来生成回答。虽然CLIP本身不直接生成图像描述，但它可以与文本生成模型（如GPT-3）结合使用，以生成与图像内容相匹配的文本描述。这种能力使CLIP在图像理解、自动标注和辅助视觉障碍人士等应用中发挥重要作用。

## 基本用途

使用CLIP算法，我们可以对海量乱七八糟的图片进行统一过滤，筛选出符合条件的图片来。譬如将10亿张图片批量地用CLIP算法过滤一道，就能得到属于同样主题的图片来。

## 代码举例

譬如我有一张图。

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/a9803af48435444cb7206033fa6af2e3~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgdWhha2Fkb3Rjb20=:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjg3NTk3ODE0NzY5MjkxMCJ9&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1730205320&x-orig-sign=osLugaIqho3r4FEvnPKUwFaQQAo%3D)

我可以写个算法在Colab运行，从而得到符合条件的概率。代码如下：

    # 安装必要的库
    !pip install git+https://github.com/openai/CLIP.git
    !pip install torch torchvision

    import torch
    import clip
    from PIL import Image
    import requests

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载CLIP模型和预处理函数
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 加载图像
    image_url = "https://res.cloudinary.com/dlxdbsdqx/image/upload/e_enhance/e_sharpen/q_auto:best/bd9l5gvapv0aonqlbtfd.png"
    image = preprocess(Image.open(requests.get(image_url, stream=True).raw)).unsqueeze(0).to(device)

    # 准备文本查询
    text = clip.tokenize(["a beautiful landscape", "a clock with gold-colored numbers on a black background"]).to(device)

    # 计算图像和文本的特征
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # 计算相似度
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("Image matches text with probabilities:", probs)

当我运行之后，得到结果为

    Image matches text with probabilities: [[0.95325935 0.04674059]]

这意味着：

1.  算法认为这张图与"a beautiful landscape"匹配的概率是95.3%；
2.  算法认为这张图与"a clock with gold-colored numbers on a black background"匹配的概率是4.6%

# 相关代码地址

*   相关代码已经开源到：<https://github.com/zgimszhd61/clip-example>
