def normalize_triplets(image_anchor, image_neighbor, image_distant):
    image_anchor = tf.cast(image_anchor, tf.float32) * (1. / 255) - 0.5
    image_neighbor = tf.cast(image_neighbor, tf.float32) * (1. / 255) - 0.5
    image_distant = tf.cast(image_distant, tf.float32) * (1. / 255) - 0.5   
    return image_anchor, image_neighbor, image_distant

def parse_function(serialized):
    IMAGE_SIZE = 128
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

    features = \
        {
            'image_anchor': tf.io.FixedLenFeature([], tf.string),
            'image_neighbor': tf.io.FixedLenFeature([], tf.string),
            'image_distant': tf.io.FixedLenFeature([], tf.string)
        }

    parsed_example = tf.io.parse_single_example(serialized=serialized,
                                             features=features)

    image_anchor = tf.decode_raw(parsed_example['image_anchor'], tf.uint8)
    image_neighbor = tf.decode_raw(parsed_example['image_neighbor'], tf.uint8)
    image_distant = tf.decode_raw(parsed_example['image_distant'], tf.uint8)

    image_anchor.set_shape((IMAGE_PIXELS))
    image_neighbor.set_shape((IMAGE_PIXELS))
    image_distant.set_shape((IMAGE_PIXELS))

    return image_anchor, image_neighbor, image_distant
