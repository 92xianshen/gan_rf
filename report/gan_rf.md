# Research on Receptive Fields of GAN

2019-07-09 矫立斌

## 摘要

- 主要探究一下rf对于GAN生成能力的影响
- 计划使用的数据集包括MNIST，CelebA和ImageNet
- 主要分为Image级别和Instance级别
- 参考文章包括Markovian Discriminator（PatchGAN）、DCGAN、WGAN-GP和BigBiGAN等
- 实验报告要记录使用的网络模型、超参数和最终结果
- 项目代号gan_rf

## 方法

- rf只有conv层和pool层才计算
- 记首层$i = 1$
- $(i + 1)^{th}$ feature map大小的计算

$$
w_{FM}^{(i + 1)} = \frac{w^{(i)} + 2 \times p^{(i + 1)} - k_{w}^{(i + 1)}}{s^{(i + 1)}} + 1 \\
h_{FM}^{(i + 1)} = \frac{h^{(i)} + 2 \times p^{(i + 1)} - k_{h}^{(i + 1)}}{s^{(i + 1)}} + 1
$$

- 经过$i^{th}$卷积层后，receptive field大小的计算
    - $i = 1$时，rf大小即为第一层卷积层kernel的大小
        $$
        w_{RF}^{(1)} = k_{w}^{(1)} \\
        h_{RF}^{(1)} = k_{h}^{(1)}
        $$
    - $i \geq 2$时，倒推至$j = 1$
        $$
        w_{RF}^{(j)} = \left\{
        \begin{aligned}
            &1, &j = i + 1 \\
            &(w_{RF}^{(j + 1)} - 1) \times s^{(j)} + k_{w}^{(j)}, &j = i, i - 1, \dots, 1 
        \end{aligned}    
        \right. \\
        h_{RF}^{(j)} = \left\{
        \begin{aligned}
            &1, &j = i + 1 \\
            &(h_{RF}^{(j + 1)} - 1) \times s^{(j)} + k_{h}^{(j)}, &j = i, i - 1, \dots, 1 
        \end{aligned}    
        \right.
        $$

## 探究内容

- rf大小影响
  - rf = 3, 5, 7等对于gan生成能力的影响
  - rf等效3, 5, 7等对于gan生成能力的影响
- 控制变量
  - 相同层数情况下，rf大小对于gan生成能力的影响
  - rf大小等效的情况下，层数对于gan生成能力的影响

## MNIST

### 模型generator
- fms: $7 \times 7 \times 256 \rightarrow 7 \times 7 \times 128 \rightarrow 14 \times 14 \times 64 \rightarrow 28 \times 28 \times 1$
- 代码

```
class Generator(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = keras.layers.Dense(
            7 * 7 * 256, use_bias=False, input_shape=(100, ))
        self.bn1 = keras.layers.BatchNormalization()
        self.lrelu1 = keras.layers.LeakyReLU()

        self.reshape1 = keras.layers.Reshape((7, 7, 256))

        self.deconv2 = keras.layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.bn2 = keras.layers.BatchNormalization()
        self.lrelu2 = keras.layers.LeakyReLU()

        self.deconv3 = keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.bn3 = keras.layers.BatchNormalization()
        self.lrelu3 = keras.layers.LeakyReLU()

        self.deconv4 = keras.layers.Conv2DTranspose(1, (5, 5), strides=(
            2, 2), padding='same', use_bias=False, activation='tanh')

    @tf.function
    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.lrelu1(x)

        x = self.reshape1(x)

        x = self.deconv2(x)
        x = self.bn2(x, training=training)
        x = self.lrelu2(x)

        x = self.deconv3(x)
        x = self.bn3(x, training=training)
        x = self.lrelu3(x)

        x = self.deconv4(x)

        return x
```

### 初始化discriminator 

- fms: $28 \times 28 \times 1 \rightarrow 14 \times 14 \times 64 \rightarrow 7 \times 7 \times 128 \rightarrow 6272 \rightarrow 1$
- rf计算
  - conv2：$(1 - 1) \times 2 + 5 = 5$
  - conv1: $(5 - 1) \times 2 + 5 = 13$
- 代码

```
class Discriminator(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = keras.layers.Conv2D(64, (5, 5), strides=(
            2, 2), padding='same', input_shape=[28, 28, 1])
        self.lrelu1 = keras.layers.LeakyReLU()
        self.dropout1 = keras.layers.Dropout(0.3)

        self.conv2 = keras.layers.Conv2D(
            128, (5, 5), strides=(2, 2), padding='same')
        self.lrelu2 = keras.layers.LeakyReLU()
        self.dropout2 = keras.layers.Dropout(.3)

        self.flat3 = keras.layers.Flatten()
        self.dense3 = keras.layers.Dense(1)

    @tf.function
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.dropout2(x)

        x = self.flat3(x)
        x = self.dense3(x)

        return x
```

### 开始实验

1. onelayer_multi_rfs
   - 一层conv
   - 在`onelayer_multi_rfs/`下
   - 代码

    ```
    class Discriminator(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # kernel size
        self.conv1 = keras.layers.Conv2D(64, (1, 1), strides=(
            1, 1), padding='same', input_shape=[28, 28, 1])
        self.lrelu1 = keras.layers.LeakyReLU()

        self.flat3 = keras.layers.Flatten()
        self.dense3 = keras.layers.Dense(1)

    @tf.function
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.lrelu1(x)

        x = self.flat3(x)
        x = self.dense3(x)

        return x
    ```
   - 结果
        
        | rf size | epochs | noise_dim | time | result | appendix |
        |:--------|:-------|:----------|:-----|:-------|:---------|
        |1x1      |50      |100        |<7s   |no      |no number |
        |3x3      |50      |100        |<7s   |yes     |not clear |
        |5x5      |50      |100        |<7s   |yes     |recognizable |
        |7x7      |50      |100        |<7s   |yes     |better    |
        |9x9      |50      |100        |~7s   |yes     |better    |
        |11x11    |50      |100        |>7s   |yes     |better    |
        |13x13    |50      |100        |7.5s  |yes     |equivalent to 13 of 2 layers |
        |28x28    |50      |100        |9.5s  |yes     |          |

   - 从视觉角度上来看，感觉除了1x1以外，其余都能够成功生成图像
   - time consumption上增长比较缓慢
   - 结论：只有一层时，rf大小不影响生成 

2. twolayers_multi_rfs
   - 二层conv，计算等效rf
   - 在`twolayers_multi_rfs/`下
   - 