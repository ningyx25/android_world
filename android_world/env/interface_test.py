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

import base64
import io
from unittest import mock

from absl.testing import absltest
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
import numpy as np
from PIL import Image
import requests


class InterfaceTest(absltest.TestCase):

  @mock.patch("time.sleep", return_value=None)
  def test_ui_stability_true(self, unused_mocked_time_sleep):
    stable_ui_elements = [representation_utils.UIElement(text="StableElement")]
    states = [
        interface.State(
            ui_elements=stable_ui_elements,
            pixels=np.empty([1, 2, 3]),
            forest=None,
        )
        for _ in range(4)
    ]
    env = interface.AsyncAndroidEnv(mock.MagicMock())
    env._get_state = mock.MagicMock(side_effect=states)

    self.assertEqual(
        env._get_stable_state(
            stability_threshold=3, sleep_duration=0.1, timeout=1
        ),
        states[2],
    )

  def test_ui_stability_false_due_to_timeout(self):
    changing_ui_elements = [
        representation_utils.UIElement(text=f"Element{i}") for i in range(10)
    ]
    env = interface.AsyncAndroidEnv(mock.MagicMock())
    states = [
        interface.State(
            ui_elements=[elem], pixels=np.empty([1, 2, 3]), forest=None
        )
        for elem in changing_ui_elements
    ]
    env._get_state = mock.MagicMock(side_effect=states)
    self.assertEqual(
        env._get_stable_state(
            stability_threshold=3, sleep_duration=0.1, timeout=0.41
        ),
        states[5],
    )

  @mock.patch("time.sleep", return_value=None)
  def test_stability_fluctuates(self, unused_mocked_time_sleep):
    env = interface.AsyncAndroidEnv(mock.MagicMock())
    fluctuating_ui_elements = (
        [representation_utils.UIElement(text="Stable")] * 2
        + [representation_utils.UIElement(text="Unstable")]
        + [representation_utils.UIElement(text="Stable")] * 3
        + [representation_utils.UIElement(text="Unstable")]
    )
    states = [
        interface.State(
            ui_elements=[elem], pixels=np.empty([1, 2, 3]), forest=None
        )
        for elem in fluctuating_ui_elements
    ]
    env._get_state = mock.MagicMock(side_effect=states)
    cur = env._get_stable_state(
        stability_threshold=3, sleep_duration=0.5, timeout=2.5
    )
    self.assertEqual(
        cur,
        states[5],
    )


class AndroidEnvClientTest(absltest.TestCase):
  """Tests for the HTTP client talking to the Android environment server."""

  BASE_URL = 'http://localhost:5000'

  def setUp(self):
    super().setUp()
    self.client = interface.AndroidEnvClient(base_url=self.BASE_URL)

  def _make_response(
      self, status_code: int = 200, json_data: dict | None = None
  ):
    """Builds a fake requests.Response-like object."""
    response = mock.MagicMock()
    response.status_code = status_code
    response.ok = status_code < 400
    response.url = f'{self.BASE_URL}/test'
    response.text = ''
    response.json.return_value = (
        json_data
        if json_data is not None
        else {'status': 'success', 'message': ''}
    )
    if response.ok:
      response.raise_for_status.return_value = None
    else:
      response.raise_for_status.side_effect = requests.HTTPError(
          f'{status_code} Server Error', response=response
      )
    return response

  @mock.patch("time.sleep", return_value=None)
  def test_request_with_retry_returns_successful_response(self, mock_sleep):
    response = self._make_response()
    send_request = mock.MagicMock(return_value=response)

    result = self.client._request_with_retry(send_request)

    self.assertIs(result, response)
    self.assertEqual(send_request.call_count, 1)
    mock_sleep.assert_not_called()

  @mock.patch("time.sleep", return_value=None)
  def test_request_with_retry_retries_transient_statuses(self, mock_sleep):
    responses = [
        self._make_response(500),
        self._make_response(503),
        self._make_response(200),
    ]
    send_request = mock.MagicMock(side_effect=responses)

    result = self.client._request_with_retry(send_request, backoff_s=1.5)

    self.assertIs(result, responses[-1])
    self.assertEqual(send_request.call_count, 3)
    self.assertEqual(
        mock_sleep.call_args_list, [mock.call(1.5), mock.call(1.5)]
    )

  @mock.patch("time.sleep", return_value=None)
  def test_request_with_retry_retries_connection_errors(self, mock_sleep):
    response = self._make_response()
    send_request = mock.MagicMock(
        side_effect=[
            requests.ConnectionError('connection refused'),
            requests.Timeout('timed out'),
            response,
        ]
    )

    result = self.client._request_with_retry(send_request, backoff_s=0.5)

    self.assertIs(result, response)
    self.assertEqual(send_request.call_count, 3)
    self.assertEqual(
        mock_sleep.call_args_list, [mock.call(0.5), mock.call(0.5)]
    )

  @mock.patch("time.sleep", return_value=None)
  def test_request_with_retry_raises_after_exhausting_attempts(
      self, mock_sleep
  ):
    responses = [self._make_response(502) for _ in range(3)]
    send_request = mock.MagicMock(side_effect=responses)

    with self.assertRaises(requests.HTTPError):
      self.client._request_with_retry(send_request)

    self.assertEqual(send_request.call_count, 3)

  @mock.patch("time.sleep", return_value=None)
  def test_request_with_retry_raises_after_repeated_connection_errors(
      self, mock_sleep
  ):
    send_request = mock.MagicMock(
        side_effect=requests.ConnectionError('connection refused')
    )

    with self.assertRaises(requests.ConnectionError):
      self.client._request_with_retry(send_request, attempts=2)

    self.assertEqual(send_request.call_count, 2)

  @mock.patch("time.sleep", return_value=None)
  def test_request_with_retry_does_not_retry_client_errors(self, mock_sleep):
    response = self._make_response(400)
    send_request = mock.MagicMock(return_value=response)

    with self.assertRaises(requests.HTTPError):
      self.client._request_with_retry(send_request)

    self.assertEqual(send_request.call_count, 1)
    mock_sleep.assert_not_called()

  @mock.patch("time.sleep", return_value=None)
  def test_request_with_retry_no_status_retries_fails_fast(self, mock_sleep):
    """With retry_statuses disabled, a 500 must not be re-sent."""
    response = self._make_response(500)
    send_request = mock.MagicMock(return_value=response)

    with self.assertRaises(requests.HTTPError):
      self.client._request_with_retry(send_request, retry_statuses=frozenset())

    self.assertEqual(send_request.call_count, 1)
    mock_sleep.assert_not_called()

  @mock.patch("time.sleep", return_value=None)
  def test_execute_action_does_not_retry_http_failures(self, mock_sleep):
    """Actions may already have executed server-side, so no HTTP retries."""
    action = json_action.JSONAction(action_type=json_action.CLICK, x=1, y=2)

    with mock.patch.object(
        interface, '_post', return_value=self._make_response(500)
    ) as mock_post:
      with self.assertRaises(requests.HTTPError):
        self.client.execute_action(action)

    mock_post.assert_called_once_with(
        f'{self.BASE_URL}/execute_action',
        json={'action_type': 'click', 'x': 1, 'y': 2},
        timeout=interface.SLOW_ENDPOINT_TIMEOUT_SEC,
    )

  @mock.patch("time.sleep", return_value=None)
  def test_execute_action_retries_connection_errors(self, mock_sleep):
    """Connection errors mean the request never reached the server."""
    action = json_action.JSONAction(action_type=json_action.NAVIGATE_HOME)
    responses = [
        requests.ConnectionError('connection refused'),
        self._make_response(json_data={'status': 'success', 'message': 'ok'}),
    ]

    with mock.patch.object(
        interface, '_post', side_effect=responses
    ) as mock_post:
      response = self.client.execute_action(action)

    self.assertEqual(
        response, interface.Response(status='success', message='ok')
    )
    self.assertEqual(mock_post.call_count, 2)

  def test_get_screenshot_decodes_base64_png(self):
    expected = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(expected).save(buffer, format='PNG')
    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    response = self._make_response(json_data={'image_b64': image_b64})

    with mock.patch.object(
        interface, '_post', return_value=response
    ) as mock_post:
      screenshot = self.client.get_screenshot()

    np.testing.assert_array_equal(screenshot, expected)
    mock_post.assert_called_once_with(
        f'{self.BASE_URL}/screenshot',
        params={'wait_to_stabilize': False},
        timeout=interface.SCREENSHOT_TIMEOUT_SEC,
    )

  def _state_response(self) -> mock.MagicMock:
    """Builds a /state response with a JPEG image and two UI elements."""
    pixels = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format='JPEG', quality=85)
    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    ui_element = representation_utils.UIElement(
        text='Settings',
        content_description='Settings',
        class_name='android.widget.TextView',
        bbox=representation_utils.BoundingBox(0.1, 0.2, 0.3, 0.4),
        bbox_pixels=representation_utils.BoundingBox(108, 216, 324, 432),
        is_clickable=True,
        is_visible=True,
        package_name='com.android.settings',
        resource_id='settings_button',
    )
    return self._make_response(
        json_data={
            'image_b64': image_b64,
            'ui_elements': [
                representation_utils.ui_element_to_dict(ui_element)
            ],
        }
    )

  def test_get_state_decodes_pixels_and_ui_elements(self):
    response = self._state_response()

    with mock.patch.object(
        interface, '_post', return_value=response
    ) as mock_post:
      state = self.client.get_state()

    self.assertEqual(state.pixels.shape, (1, 2, 3))
    self.assertIsNone(state.forest)  # No forest in Docker mode.
    self.assertEqual(state.auxiliaries, {})
    self.assertLen(state.ui_elements, 1)
    element = state.ui_elements[0]
    self.assertEqual(element.text, 'Settings')
    self.assertEqual(element.class_name, 'android.widget.TextView')
    self.assertEqual(element.package_name, 'com.android.settings')
    self.assertEqual(element.resource_id, 'settings_button')
    self.assertTrue(element.is_clickable)
    self.assertEqual(element.bbox_pixels.x_min, 108)
    self.assertEqual(element.bbox_pixels.center, (162.0, 378.0))
    mock_post.assert_called_once_with(
        f'{self.BASE_URL}/state',
        params={'wait_to_stabilize': False},
        timeout=interface.SCREENSHOT_TIMEOUT_SEC,
    )

  def test_get_state_wait_to_stabilize_uses_slower_timeout(self):
    response = self._state_response()

    with mock.patch.object(
        interface, '_post', return_value=response
    ) as mock_post:
      self.client.get_state(wait_to_stabilize=True)

    mock_post.assert_called_once_with(
        f'{self.BASE_URL}/state',
        params={'wait_to_stabilize': True},
        timeout=interface.SLOW_ENDPOINT_TIMEOUT_SEC,
    )

  def test_get_state_retries_transient_failures(self):
    responses = [
        self._make_response(503),
        self._state_response(),
    ]

    with mock.patch("time.sleep", return_value=None), mock.patch.object(
        interface, '_post', side_effect=responses
    ) as mock_post:
      state = self.client.get_state()

    self.assertLen(state.ui_elements, 1)
    self.assertEqual(mock_post.call_count, 2)


if __name__ == "__main__":
  absltest.main()
