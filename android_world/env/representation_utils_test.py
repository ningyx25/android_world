# Copyright 2026 The android_world Authors.
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

import dataclasses
import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from android_world.env import representation_utils
import numpy as np


@dataclasses.dataclass(frozen=True)
class BoundsInScreen:
  left: int
  right: int
  top: int
  bottom: int


class TestAccessibilityNodeToUIElement(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='rectangle_to_rectangle_normalization',
          node_bounds=BoundsInScreen(0, 150, 0, 100),
          screen_size=(500, 500),
          expected_normalized_bbox=representation_utils.BoundingBox(
              0.0, 0.3, 0.0, 0.2
          ),
      ),
      dict(
          testcase_name='square_to_square_normalization',
          node_bounds=BoundsInScreen(100, 200, 100, 200),
          screen_size=(1000, 1000),
          expected_normalized_bbox=representation_utils.BoundingBox(
              0.1, 0.2, 0.1, 0.2
          ),
      ),
      dict(
          testcase_name='square_to_rectangle_normalization',
          node_bounds=BoundsInScreen(0, 100, 0, 100),
          screen_size=(1000, 500),
          expected_normalized_bbox=representation_utils.BoundingBox(
              0.0, 0.1, 0.0, 0.2
          ),
      ),
      dict(
          testcase_name='no_change_square_normalization',
          node_bounds=BoundsInScreen(0, 100, 0, 100),
          screen_size=(100, 100),
          expected_normalized_bbox=representation_utils.BoundingBox(
              0.0, 1.0, 0.0, 1.0
          ),
      ),
      dict(
          testcase_name='no_change_rectangle_normalization',
          node_bounds=BoundsInScreen(0, 200, 0, 100),
          screen_size=(200, 100),
          expected_normalized_bbox=representation_utils.BoundingBox(
              0.0, 1.0, 0.0, 1.0
          ),
      ),
      dict(
          testcase_name='normalization_causing_dimensions_to_grow',
          node_bounds=BoundsInScreen(0, 50, 0, 50),
          screen_size=(200, 200),
          expected_normalized_bbox=representation_utils.BoundingBox(
              0.0, 0.25, 0.0, 0.25
          ),
      ),
      dict(
          testcase_name='zero_size_bbox_normalization',
          node_bounds=BoundsInScreen(0, 0, 0, 0),
          screen_size=(100, 100),
          expected_normalized_bbox=representation_utils.BoundingBox(
              0.0, 0.0, 0.0, 0.0
          ),
      ),
      dict(
          testcase_name='no_normalization',
          node_bounds=BoundsInScreen(10, 20, 11, 13),
          screen_size=None,
          expected_normalized_bbox=None,
      ),
  )
  def test_normalize_bboxes(
      self, node_bounds, screen_size, expected_normalized_bbox
  ):
    node = mock.MagicMock()
    node.bounds_in_screen = node_bounds

    ui_element = representation_utils.accessibility_node_to_ui_element(
        node, screen_size
    )
    self.assertEqual(ui_element.bbox_pixels.x_min, node_bounds.left)
    self.assertEqual(ui_element.bbox_pixels.x_max, node_bounds.right)
    self.assertEqual(ui_element.bbox_pixels.y_min, node_bounds.top)
    self.assertEqual(ui_element.bbox_pixels.y_max, node_bounds.bottom)

    if screen_size is not None:
      ui_element.bbox = representation_utils._normalize_bounding_box(
          ui_element.bbox_pixels, screen_size
      )
    self.assertEqual(ui_element.bbox, expected_normalized_bbox)


class UiElementSerializationTest(absltest.TestCase):
  """Round-trip tests for the JSON transport used by the /state endpoint."""

  def _make_element(self) -> representation_utils.UIElement:
    return representation_utils.UIElement(
        text='Settings',
        content_description='Open settings',
        class_name='android.widget.TextView',
        bbox=representation_utils.BoundingBox(0.1, 0.2, 0.3, 0.4),
        bbox_pixels=representation_utils.BoundingBox(108, 216, 324, 432),
        hint_text='hint',
        is_checked=False,
        is_checkable=True,
        is_clickable=True,
        is_editable=False,
        is_enabled=True,
        is_focused=False,
        is_focusable=True,
        is_long_clickable=False,
        is_scrollable=False,
        is_selected=False,
        is_visible=True,
        package_name='com.android.settings',
        resource_name='settings_button_view',
        tooltip='tooltip',
        resource_id='settings_button',
        metadata={'source': 'test'},
    )

  def test_round_trip_preserves_all_fields(self):
    element = self._make_element()

    restored = representation_utils.ui_element_from_dict(
        representation_utils.ui_element_to_dict(element)
    )

    self.assertEqual(restored, element)

  def test_dict_is_json_serializable(self):
    """The server json-encodes this dict; numpy/proto scalars must not leak."""
    element = self._make_element()
    element.bbox_pixels = representation_utils.BoundingBox(
        np.int64(1), np.int64(2), np.int64(3), np.int64(4)
    )

    payload = json.dumps(representation_utils.ui_element_to_dict(element))

    self.assertIn('Settings', payload)

  def test_none_bounding_box_round_trips(self):
    element = representation_utils.UIElement(text='no bbox')

    restored = representation_utils.ui_element_from_dict(
        representation_utils.ui_element_to_dict(element)
    )

    self.assertIsNone(restored.bbox)
    self.assertIsNone(restored.bbox_pixels)

  def test_from_dict_tolerates_unknown_keys(self):
    """Client must not break when the server sends extra fields."""
    element = self._make_element()
    data = representation_utils.ui_element_to_dict(element)
    data['future_field'] = 'ignored'

    restored = representation_utils.ui_element_from_dict(data)

    self.assertEqual(restored, element)


if __name__ == '__main__':
  absltest.main()
