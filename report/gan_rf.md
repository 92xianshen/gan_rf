# Research on Receptive Fields of GAN

2019-07-09 矫立斌

## 摘要

- 主要探究一下RF对于GAN生成能力的影响
- 计划使用的数据集包括MNIST，CelebA和ImageNet
- 主要分为Image级别和Instance级别
- 参考文章包括Markovian Discriminator（PatchGAN）、DCGAN、WGAN-GP和BigBiGAN等
- 实验报告要记录使用的网络模型、超参数和最终结果
- 项目代号gan_rf

## MNIST

### 模型generator - feature maps: $7 \times 7 \times 256 \rightarrow 7 \times 7 \times 128 \rightarrow 14 \times 14 \times 64 \rightarrow 28 \times 28 \times 1$

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

### 初始化discriminator - feature maps: $28 \times 28 \times 1 \rightarrow 14 \times 14 \times 64 \rightarrow 7 \times 7 \times 128 \rightarrow 6272 \rightarrow 1$

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