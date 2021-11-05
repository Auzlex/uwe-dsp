"""
    This script is used to test TF
    fundemental operations.
"""
import tensorflow as tf
from tensorflow._api.v2 import random

"""
    Scalar: a single number
    Vector: a set of numbers indicating direction and intensity such as wind speed
    Matrix: a d-dimensional array of numbers
    Tensor: an n-dimensional array of numbers (where n can be any number, a 0-dimensional tensor is a scalar, a 1-dimensional tensor is a vector)
"""

"""
    Method of creating a tensors:
    1. Using tf.constant()
    2. Using tf.Variable()
    3. Using tf.placeholder()
"""

"""matrix 2d array of numbers"""
# example_matrix = tf.constant(
#     [ 
#         [1, 2, 3], 
#         [4, 5, 6],
#         [8, 4, 2]
#     ],
#     dtype=tf.float32
#     )

#print(example_matrix)
#print(example_matrix.ndim)

"""tensor is an n-dimensional array"""
# tensor = tf.constant(
#     [
#         [
#             [ 1,2,3 ],
#             [ 4,5,6 ]
#         ],
#         [
#             [ 7,8,9 ],
#             [ 10,11,12 ]
#         ],
#         [
#             [ 13,14,15 ],
#             [ 16,17,18 ]
#         ]
#     ]
# )

#print(tensor)
#print(tensor.ndim)

"""
    tf.Variable
"""

# changeable_tensor = tf.Variable([10,7])
# unchangeable_tensor = tf.constant([10,7])

# #print(changeable_tensor, "\n", unchangeable_tensor)
# print(changeable_tensor)

# # changeable_tensor[0] = 7, will not work we must use assign
# changeable_tensor[0].assign(7)

# print(changeable_tensor)
"""
    NOTE: Rarely in practice will you need to decide weather to use tf.Variable or tf.constant to create tensors.
    as tensorflow does this for you, However

    if in doubt use tf.constant and change it later if needed
"""

"""
    Creating random tensors
"""
random_1 = tf.random.Generator.from_seed(42) # we set a seed to get reproducible results
random_1 = random.normal(shape=(3,2))
print(random_1)

random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3,2))
print(random_2)