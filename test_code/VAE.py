import tensorflow as tf


class CVAE(tf.keras.Model):
    """Convolutional varioational autoencoder."""

    def __init__(self, latent_dim, spatial_dist, realizations=1):
        super(CVAE, self).__init__()
        self.realizations = realizations
        self.spatial_dist = spatial_dist
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(16, 16, realizations)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, activation="relu"),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim)
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, padding="same", activation="relu"),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, padding="same")
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(256, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        log_nugget, spatial_range = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return log_nugget, spatial_range

    def reparametrize(self, log_nugget, spatial_range):
        eps = tf.random.normal(256)
        covariance_mat = tf.exp(-self.spatial_dist / spatial_range) + tf.exp(log_nugget) * tf.eye(256)
        return covariance_mat @ eps

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


optimizer = tf.keras.optimizers.Adam()


def log_normal_pdf():
    pass

