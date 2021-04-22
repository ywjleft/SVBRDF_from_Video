# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Image warping using per-pixel flow vectors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf

def _interpolate_bilinear(grid,
                          query_points,
                          name='interpolate_bilinear',
                          indexing='ij', constvalue=-3.402823e38):
  """Similar to Matlab's interp2 function.

  Finds values for query points on a grid using bilinear interpolation.

  Args:
    grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
    query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
    name: a name for the operation (optional).
    indexing: whether the query points are specified as row and column (ij),
      or Cartesian coordinates (xy).

  Returns:
    values: a 3-D `Tensor` with shape `[batch, N, channels]`

  Raises:
    ValueError: if the indexing mode is invalid, or if the shape of the inputs
      invalid.
  """
  if indexing != 'ij' and indexing != 'xy':
    raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

  with ops.name_scope(name):
    grid = ops.convert_to_tensor(grid)
    query_points = ops.convert_to_tensor(query_points)
    shape = grid.get_shape().as_list()
    if len(shape) != 4:
      msg = 'Grid must be 4 dimensional. Received size: '
      raise ValueError(msg + str(grid.get_shape()))

    _, height, width, channels = shape
    batch_size = tf.shape(grid)[0]
    query_type = query_points.dtype
    grid_type = grid.dtype

    if (len(query_points.get_shape()) != 3 or
        query_points.get_shape()[2].value != 2):
      msg = ('Query points must be 3 dimensional and size 2 in dim 2. Received '
             'size: ')
      raise ValueError(msg + str(query_points.get_shape()))

    _, num_queries, _ = query_points.get_shape().as_list()

    if height < 2 or width < 2:
      msg = 'Grid must be at least batch_size x 2 x 2 in size. Received size: '
      raise ValueError(msg + str(grid.get_shape()))

    grid = tf.pad(grid, [[0,0],[2,2],[2,2],[0,0]], 'CONSTANT', constant_values=constvalue) #smallest value of float32

    alphas = []
    floors = []
    ceils = []

    index_order = [0, 1] if indexing == 'ij' else [1, 0]
    unstacked_query_points = array_ops.unstack(query_points, axis=2)

    for dim in index_order:
      with ops.name_scope('dim-' + str(dim)):
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = shape[dim + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = math_ops.cast(size_in_indexing_dimension, query_type)
        min_floor = constant_op.constant(-2.0, dtype=query_type)
        floor = math_ops.minimum(
            math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
        int_floor = math_ops.cast(floor + 2, dtypes.int32)
        floors.append(int_floor)

        ceil = math_ops.minimum(
            math_ops.maximum(min_floor, math_ops.ceil(queries)), max_floor)
        int_ceil = math_ops.cast(ceil + 2, dtypes.int32)
        ceils.append(int_ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = math_ops.cast(queries - floor, grid_type)
        min_alpha = constant_op.constant(0.0, dtype=grid_type)
        max_alpha = constant_op.constant(1.0, dtype=grid_type)
        alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = array_ops.expand_dims(alpha, 2)
        alphas.append(alpha)
    '''
    if batch_size * height * width > np.iinfo(np.int32).max / 8:
      error_msg = """The image size or batch size is sufficiently large
                     that the linearized addresses used by array_ops.gather
                     may exceed the int32 limit."""
      raise ValueError(error_msg)
    '''

    flattened_grid = array_ops.reshape(grid,
                                       [batch_size * (height+4) * (width+4), channels])
    batch_offsets = array_ops.reshape(
        math_ops.range(batch_size) * (height+4) * (width+4), [batch_size, 1])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(y_coords, x_coords, name):
      with ops.name_scope('gather-' + name):
        linear_coordinates = batch_offsets + y_coords * (width+4) + x_coords
        gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
        return array_ops.reshape(gathered_values,
                                 [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], 'top_left')
    top_right = gather(floors[0], ceils[1], 'top_right')
    bottom_left = gather(ceils[0], floors[1], 'bottom_left')
    bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

    # now, do the actual interpolation
    with ops.name_scope('interpolate'):
      condition = tf.logical_or(tf.logical_or(tf.logical_or(tf.less(top_left, -3e38), tf.less(top_right, -3e38)), tf.less(bottom_left, -3e38)), tf.less(bottom_right, -3e38))
      interp = tf.where(condition, tf.fill(tf.shape(condition), constvalue), alphas[0] * (alphas[1] * bottom_right + (1 - alphas[1]) * bottom_left) + (1 - alphas[0]) * (alphas[1] * top_right + (1 - alphas[1]) * top_left))

    return interp


def dense_image_warp(image, flow, constvalue=-3.402823e38, name='dense_image_warp'):
  """Image warping using per-pixel flow vectors.

  Apply a non-linear warp to the image, where the warp is specified by a dense
  flow field of offset vectors that define the correspondences of pixel values
  in the output image back to locations in the  source image. Specifically, the
  pixel value at output[b, j, i, c] is
  images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].

  The locations specified by this formula do not necessarily map to an int
  index. Therefore, the pixel value is obtained by bilinear
  interpolation of the 4 nearest pixels around
  (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
  of the image, we use the nearest pixel values at the image boundary.


  Args:
    image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
    flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
    name: A name for the operation (optional).

    Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
    and do not necessarily have to be the same type.

  Returns:
    A 4-D float `Tensor` with shape`[batch, height, width, channels]`
      and same type as input image.

  Raises:
    ValueError: if height < 2 or width < 2 or the inputs have the wrong number
                of dimensions.
  """
  with ops.name_scope(name):
    flow = flow[...,::-1]
    _, height, width, channels = image.get_shape().as_list()
    batch_size = tf.shape(image)[0]
    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    grid_x, grid_y = array_ops.meshgrid(
        math_ops.range(width), math_ops.range(height))
    stacked_grid = math_ops.cast(
        array_ops.stack([grid_y, grid_x], axis=2), flow.dtype)
    batched_grid = array_ops.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid - flow
    query_points_flattened = array_ops.reshape(query_points_on_grid,
                                               [batch_size, height * width, 2])
    # Compute values at the query points, then reshape the result back to the
    # image grid.
    interpolated = _interpolate_bilinear(image, query_points_flattened, constvalue=constvalue)
    interpolated = array_ops.reshape(interpolated,
                                     [batch_size, height, width, channels])
    return interpolated
