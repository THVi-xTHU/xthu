# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Numpy BoxList classes and functions."""

import numpy as np

from utility import *

class BoxList(object):
  """Box collection.

  BoxList represents a list of bounding boxes as numpy array, where each
  bounding box is represented as a row of 4 numbers,
  [y_min, x_min, y_max, x_max].  It is assumed that all bounding boxes within a
  given list correspond to a single image.

  Optionally, users can add additional related fields (such as
  objectness/classification scores).
  """

  def __init__(self, data=None):
    """Constructs box collection.

    Args:
      data: a numpy array of shape [N, 4] representing box coordinates

    Raises:
      ValueError: if bbox data is not a numpy array
      ValueError: if invalid dimensions for bbox data
    """
    self.shape = None
    if data is None:
      self.data = {'boxes': np.empty((0,4))}
      return 
    if isinstance(data, dict):
      self.data = {
        k: np.asarray(v) for k, v in data.items()
      }
      self.data['boxes'] = self.data['boxes'].reshape(-1, 4)
    else:
      self.data = {'boxes': np.array(data).reshape(-1, 4)}
    data = self.data['boxes']
    if not isinstance(data, np.ndarray):
      raise ValueError('data must be a numpy array, got %s instead.' % type(data))
    if (len(data.shape) != 2 or data.shape[1] != 4) and (data.shape[0] != 0):
      raise ValueError('Invalid dimensions for box data, except (?, 4) dimensions, got {0} instead.'.format(data.shape))
    if not self._is_valid_boxes(data):
      raise ValueError('Invalid box data. data must be a numpy array of '
                       'N*[y_min, x_min, y_max, x_max]')


  def num_boxes(self):
    """Return number of boxes held in collections."""
    return self.data['boxes'].shape[0]

  def get_extra_fields(self):
    """Return all non-box fields."""
    return [k for k in self.data.keys() if k != 'boxes']

  def has_field(self, field):
    return field in self.data

  def add_field(self, field, field_data):
    """Add data to a specified field.

    Args:
      field: a string parameter used to speficy a related field to be accessed.
      field_data: a numpy array of [N, ...] representing the data associated
          with the field.
    Raises:
      ValueError: if the field is already exist or the dimension of the field
          data does not matches the number of boxes.
    """
    field_data = np.asarray(field_data)
    if self.has_field(field):
      raise ValueError('Field ' + field + 'already exists')
    if len(field_data.shape) < 1 or field_data.shape[0] != self.num_boxes():
      raise ValueError('Invalid dimensions for field data')
    self.data[field] = field_data

  def add_field_data(self, field, data):

    field_data = np.asarray(data).reshape(1,-1)
    if not self.has_field(field):
      self.data[field] = field_data 
    else:
      self.data[field] = np.concatenate((self.data[field], field_data))

  def get(self):
    """Convenience function for accesssing box coordinates.

    Returns:
      a numpy array of shape [N, 4] representing box corners
    """
    return self.get_field('boxes').astype(int)

  def get_field(self, field):
    """Accesses data associated with the specified field in the box collection.

    Args:
      field: a string parameter used to speficy a related field to be accessed.

    Returns:
      a numpy 1-d array representing data of an associated field

    Raises:
      ValueError: if invalid field
    """
    if not self.has_field(field):
      return np.array([])
      #raise ValueError('field {} does not exist'.format(field))
    return self.data[field]

  def get_coordinates(self):
    """Get corner coordinates of boxes.

    Returns:
     a list of 4 1-d numpy arrays [y_min, x_min, y_max, x_max]
    """
    box_coordinates = self.get()
    y_min = box_coordinates[:, 0]
    x_min = box_coordinates[:, 1]
    y_max = box_coordinates[:, 2]
    x_max = box_coordinates[:, 3]
    return [y_min, x_min, y_max, x_max]

  def get_size_boxes(self):
    return two_point2point_size_array(self.get())

  def get_specific_data(self, field, query):
    keep = self.get_field(field) == query
    new_data = {f: d[keep] for f, d in self.data.items()}
    return BoxList(new_data)

  def exclude_specific_data(self, field, query):
    keep = self.get_field(field) != query
    new_data = {f: d[keep] for f, d in self.data.items()}
    return BoxList(new_data)

  def keep_indices(self, idx):
    
    for k, v in self.data.items():
      try:
        self.data[k] = v[idx]
      except:
        import pdb
        pdb.set_trace()
   

  def get_ratio_boxes(self):
    pass


  def _is_valid_boxes(self, data):
    """Check whether data fullfills the format of N*[ymin, xmin, ymax, xmin].

    Args:
      data: a numpy array of shape [N, 4] representing box coordinates

    Returns:
      a boolean indicating whether all ymax of boxes are equal or greater than
          ymin, and all xmax of boxes are equal or greater than xmin.
    """
    if data.shape[0] > 0:
      for i in range(data.shape[0]):
        if data[i, 0] > data[i, 2] or data[i, 1] > data[i, 3]:
          print(data[i, :])
          return False
    return True

  def __iter__(self):
    for i in range(self.num_boxes()):
      yield {k: v[i] for k, v in self.data.items()}
