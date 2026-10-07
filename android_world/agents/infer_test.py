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
import contextlib
import io
import json
import os
import time
from unittest import mock

from absl.testing import absltest
from android_world.agents import infer
import google.ai.generativelanguage as glm
import google.generativeai as genai
from google.generativeai.types import answer_types
from google.generativeai.types import generation_types
import numpy as np
from PIL import Image
import requests


class InferTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_post = mock.patch.object(requests, "post").start()
    self.mock_sleep = mock.patch.object(time, "sleep").start()
    os.environ["OPENAI_API_KEY"] = "fake_api_key"
    os.environ["GCP_API_KEY"] = "fake_api_key"

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  @mock.patch.object(genai.GenerativeModel, "generate_content")
  def test_gemini_gcp(self, mock_generate_content):
    mock_generate_content.return_value = (
        generation_types.GenerateContentResponse.from_response(
            glm.GenerateContentResponse({
                "candidates": (
                    [{"content": {"parts": [{"text": "fake response"}]}}]
                )
            })
        )
    )
    llm = infer.GeminiGcpWrapper(model_name="some_gemini_model")
    text_output, is_safe, _ = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, "fake response")
    self.assertEqual(is_safe, True)

  @mock.patch.object(genai.GenerativeModel, "generate_content")
  def test_gemini_gcp_error(self, mock_generate_content):
    mock_generate_content.return_value = (
        generation_types.GenerateContentResponse.from_response(
            glm.GenerateContentResponse(
                {"candidates": [{"content": {"parts": []}}]}
            )
        )
    )
    llm = infer.GeminiGcpWrapper(model_name="some_gemini_model")
    text_output, is_safe, output = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, infer.ERROR_CALLING_LLM)
    self.assertIsNone(is_safe)
    self.assertIsNone(output)

  @mock.patch.object(genai.GenerativeModel, "generate_content")
  def test_gemini_gcp_no_candidates(self, mock_generate_content):
    mock_generate_content.return_value = (
        generation_types.GenerateContentResponse.from_response(
            glm.GenerateContentResponse({"candidates": []})
        )
    )
    llm = infer.GeminiGcpWrapper(model_name="some_gemini_model")
    text_output, is_safe, output = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, infer.ERROR_CALLING_LLM)
    self.assertIsNone(is_safe)
    self.assertIsNone(output)

  @mock.patch.object(genai.GenerativeModel, "generate_content")
  def test_gemini_gcp_unsafe(self, mock_generate_content):
    mock_generate_content.return_value = (
        generation_types.GenerateContentResponse.from_response(
            glm.GenerateContentResponse({
                "candidates": (
                    [{
                        "content": {"parts": []},
                        "finish_reason": answer_types.FinishReason.SAFETY,
                    }]
                )
            })
        )
    )
    llm = infer.GeminiGcpWrapper(model_name="some_gemini_model")
    text_output, is_safe, _ = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, infer.ERROR_CALLING_LLM)
    self.assertEqual(is_safe, False)

  def test_gpt4v(self):
    llm = infer.Gpt4Wrapper(model_name="gpt-4-turbo-2024-04-09")
    mock_200_response = requests.Response()
    mock_200_response.status_code = 200
    mock_200_response._content = (
        b'{"choices": [{"message": {"content": "fake response"}}]}'
    )
    self.mock_post.return_value = mock_200_response

    text_output, _, _ = llm.predict_mm("fake prompt", [])
    self.assertEqual(text_output, "fake response")

  def test_gpt4v_retry(self):
    gpt4v = infer.Gpt4Wrapper(model_name="gpt-4-turbo-2024-04-09")

    mock_429_response = requests.Response()
    mock_429_response.status_code = 429
    mock_429_response._content = (
        b'{"error": {"message": "Error 429: rate limit reached."}}'
    )

    mock_200_response = requests.Response()
    mock_200_response.status_code = 200
    mock_200_response._content = (
        b'{"choices": [{"message": {"content": "ok."}}]}'
    )
    self.mock_post.side_effect = [mock_429_response, mock_200_response]

    gpt4v.predict_mm("fake prompt", [])
    self.mock_sleep.assert_called_once()


class TypeSafeJevWrapperTest(absltest.TestCase):
  """Tests the TypeSafe Jev wrapper, with no network access."""

  def setUp(self):
    super().setUp()
    self.mock_post = mock.patch.object(requests, "post").start()
    os.environ["TYPESAFE_API_KEY"] = "fake-typesafe-key"
    os.environ.pop("TYPESAFE_MODEL", None)

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()
    os.environ.pop("TYPESAFE_API_KEY", None)
    os.environ.pop("TYPESAFE_MODEL", None)

  def _response(self, status_code: int, body: bytes) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = body
    return response

  def _success_body(self) -> bytes:
    return (
        b'{"model": "jev-test", "answers": {"operation": {"type": "choice",'
        b' "choice": "DONE", "probabilities": {"DONE": 1.0}, "confidence":'
        b' 1.0}}, "usage": {"input_tokens": 1, "output_tokens": 1}}'
    )

  def test_missing_api_key_raises(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      with self.assertRaisesRegex(RuntimeError, 'TypeSafe API key not set'):
        infer.TypeSafeJevWrapper()

  def test_empty_api_key_raises(self):
    with mock.patch.dict(os.environ, {'TYPESAFE_API_KEY': '   '}):
      with self.assertRaisesRegex(RuntimeError, 'TypeSafe API key not set'):
        infer.TypeSafeJevWrapper()

  def test_default_model(self):
    wrapper = infer.TypeSafeJevWrapper()
    self.assertEqual(wrapper.model_name, 'jev-latest')

  def test_model_from_environment(self):
    with mock.patch.dict(os.environ, {'TYPESAFE_MODEL': 'jev-env'}):
      wrapper = infer.TypeSafeJevWrapper()
    self.assertEqual(wrapper.model_name, 'jev-env')

  def test_explicit_model_wins(self):
    with mock.patch.dict(os.environ, {'TYPESAFE_MODEL': 'jev-env'}):
      wrapper = infer.TypeSafeJevWrapper(model_name='jev-explicit')
    self.assertEqual(wrapper.model_name, 'jev-explicit')

  def test_successful_request(self):
    self.mock_post.return_value = self._response(200, self._success_body())
    wrapper = infer.TypeSafeJevWrapper()

    request = {'state': {'goal': 'g'}, 'questions': {'q': {}}}
    text, is_safe, raw = wrapper.predict_jev(request)

    self.assertIsNone(is_safe)
    self.assertEqual(raw, json.loads(text))
    self.assertEqual(raw['answers']['operation']['choice'], 'DONE')

    _, kwargs = self.mock_post.call_args
    self.assertEqual(
        self.mock_post.call_args.args[0], infer.TypeSafeJevWrapper.ENDPOINT
    )
    self.assertEqual(
        kwargs['headers']['Authorization'], 'Bearer fake-typesafe-key'
    )
    self.assertEqual(kwargs['timeout'], 30.0)
    self.assertEqual(kwargs['json']['model'], 'jev-latest')
    self.assertEqual(kwargs['json']['state'], {'goal': 'g'})
    self.assertEqual(kwargs['json']['questions'], {'q': {}})

  def test_http_error_is_not_retried(self):
    self.mock_post.return_value = self._response(500, b'{"error": "boom"}')
    wrapper = infer.TypeSafeJevWrapper()

    text, is_safe, raw = wrapper.predict_jev({'state': {}, 'questions': {}})

    self.assertEqual(text, infer.ERROR_CALLING_LLM)
    self.assertEqual(is_safe, False)
    self.assertIsNone(raw)
    self.mock_post.assert_called_once()

  def test_invalid_json(self):
    self.mock_post.return_value = self._response(200, b'not json')
    wrapper = infer.TypeSafeJevWrapper()

    text, is_safe, raw = wrapper.predict_jev({'state': {}, 'questions': {}})

    self.assertEqual(text, infer.ERROR_CALLING_LLM)
    self.assertEqual(is_safe, False)
    self.assertIsNone(raw)

  def test_missing_answers(self):
    self.mock_post.return_value = self._response(200, b'{"model": "jev"}')
    wrapper = infer.TypeSafeJevWrapper()

    text, is_safe, raw = wrapper.predict_jev({'state': {}, 'questions': {}})

    self.assertEqual(text, infer.ERROR_CALLING_LLM)
    self.assertEqual(is_safe, False)
    self.assertIsNone(raw)

  def test_transport_error(self):
    self.mock_post.side_effect = requests.Timeout('timed out')
    wrapper = infer.TypeSafeJevWrapper()

    text, is_safe, raw = wrapper.predict_jev({'state': {}, 'questions': {}})

    self.assertEqual(text, infer.ERROR_CALLING_LLM)
    self.assertEqual(is_safe, False)
    self.assertIsNone(raw)
    self.mock_post.assert_called_once()

  def test_error_redacts_api_key(self):
    self.mock_post.return_value = self._response(
        401, b'{"error": "invalid key fake-typesafe-key"}'
    )
    wrapper = infer.TypeSafeJevWrapper()

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      wrapper.predict_jev({'state': {}, 'questions': {}})

    output = stdout.getvalue()
    self.assertNotIn('fake-typesafe-key', output)
    self.assertIn('[redacted]', output)



class MultimodalJevWrapperTest(absltest.TestCase):
  '''Tests the vision half of the TypeSafe wrapper, without the network.'''

  def setUp(self):
    super().setUp()
    self.mock_post = mock.patch.object(requests, 'post').start()
    self.mock_get = mock.patch.object(requests, 'get').start()
    os.environ['TYPESAFE_API_KEY'] = 'fake-typesafe-key'
    os.environ['TYPESAFE_MM_ENDPOINT'] = 'http://vision.local/decide'

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()
    os.environ.pop('TYPESAFE_API_KEY', None)
    os.environ.pop('TYPESAFE_MM_ENDPOINT', None)
    os.environ.pop('TYPESAFE_ENDPOINT', None)

  def _response(self, status_code, body):
    response = requests.Response()
    response.status_code = status_code
    response._content = body
    return response

  def _images(self):
    return [
        infer.JevImage(kind='history', pixels=np.zeros((30, 40, 3), np.uint8)),
        infer.JevImage(kind='current', pixels=np.zeros((20, 20, 3), np.uint8)),
    ]

  def _body(self):
    payload = {
        'model': 'jev2',
        'answers': {'done_goal': {'type': 'noul', 'noul': 0.25,
                    'confidence': 0.75}},
        'usage': {'input_tokens': 4},
    }
    return json.dumps(payload).encode('utf-8')

  def test_images_are_the_bare_base64_list_in_prompt_order(self):
    self.mock_post.return_value = self._response(200, self._body())
    self.mock_get.return_value = self._response(
        200, json.dumps({'models': [{'description': 'Clef head'}]}).encode()
    )
    wrapper = infer.TypeSafeJevWrapper()

    text, is_safe, raw = wrapper.predict_jev_mm(
        {'state': {'goal': 'g'}, 'questions': {'done_goal': {}}},
        self._images(),
    )

    _, kwargs = self.mock_post.call_args
    self.assertEqual(
        self.mock_post.call_args.args[0], 'http://vision.local/decide'
    )
    sent = kwargs['json']['images']
    self.assertTrue(all(isinstance(image, str) for image in sent))
    first = base64.b64decode(sent[0])
    self.assertEqual(first[:4], bytes([137, 80, 78, 71]))
    with io.BytesIO(first) as handle:
      # A history frame is grown to the 512^2 lattice it was trained on.
      self.assertEqual(Image.open(handle).size, (588, 448))
    self.assertEqual(raw['answers']['done_goal']['noul'], 0.25)
    self.assertIsNone(is_safe)
    self.assertEqual(text, json.dumps(raw))

  def test_http_error_is_not_retried(self):
    self.mock_post.return_value = self._response(
        500, json.dumps({'error': 'boom'}).encode('utf-8')
    )
    wrapper = infer.TypeSafeJevWrapper()

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      result = wrapper.predict_jev_mm(
          {'state': {}, 'questions': {}}, self._images()
      )

    self.assertEqual(result, (infer.ERROR_CALLING_LLM, False, None))
    self.mock_post.assert_called_once()

  def test_the_text_endpoint_follows_the_environment(self):
    with mock.patch.dict(
        os.environ, {'TYPESAFE_ENDPOINT': 'http://text.local/decide'}
    ):
      wrapper = infer.TypeSafeJevWrapper()
    self.assertEqual(wrapper.endpoint, 'http://text.local/decide')
    self.mock_post.return_value = self._response(200, self._body())
    wrapper.predict_jev({'state': {}, 'questions': {}})
    self.assertEqual(
        self.mock_post.call_args.args[0], 'http://text.local/decide'
    )

  def test_png_survives_the_encoding_round_trip(self):
    pixels = np.zeros((4, 6, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    decoded = Image.open(
        io.BytesIO(base64.b64decode(infer.array_to_png_b64(pixels)))
    )
    self.assertEqual(decoded.mode, 'RGB')
    self.assertEqual(decoded.size, (6, 4))
    self.assertEqual(np.asarray(decoded)[0, 0].tolist(), [255, 0, 0])

  def _card(self, description):
    body = json.dumps({'models': [{'description': description}]}).encode()
    self.mock_get.return_value = self._response(200, body)

  def test_a_text_only_card_is_warned_about_once(self):
    self._card('Kev pointer head on Qwen3.5-0.8B')
    self.mock_post.return_value = self._response(200, self._body())
    wrapper = infer.TypeSafeJevWrapper()
    body = {'state': 's', 'questions': {'done_goal': {}}}

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      wrapper.predict_jev_mm(body, self._images())
      wrapper.predict_jev_mm(body, self._images())

    self.assertIn('does not look like a vision checkpoint', stdout.getvalue())
    self.assertEqual(self.mock_get.call_count, 1)

  def test_a_tall_frame_is_measured_out_by_kind(self):
    self._card('Clef joint-schema head on Qwen3.5')
    self.mock_post.return_value = self._response(200, self._body())
    tall = np.zeros((2400, 1080, 3), dtype=np.uint8)
    images = [
        infer.JevImage(kind='history', pixels=tall),
        infer.JevImage(kind='current', pixels=tall),
        infer.JevImage(kind='marked', pixels=tall),
    ]

    infer.TypeSafeJevWrapper().predict_jev_mm(
        {'state': 's', 'questions': {'q': {}}}, images
    )

    _, kwargs = self.mock_post.call_args
    sizes = [
        Image.open(io.BytesIO(base64.b64decode(image))).size
        for image in kwargs['json']['images']
    ]
    self.assertEqual(sizes[1], sizes[2])
    self.assertLess(sizes[0][0] * sizes[0][1], sizes[1][0] * sizes[1][1])
    for width, height in sizes:
      self.assertEqual(width % infer.JEV_PIXEL_PATCH, 0)
      self.assertEqual(height % infer.JEV_PIXEL_PATCH, 0)

  def test_a_vision_card_says_nothing(self):
    self._card('Clef joint-schema head on Qwen3.5')
    self.mock_post.return_value = self._response(200, self._body())

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      infer.TypeSafeJevWrapper().predict_jev_mm(
          {'state': 's', 'questions': {'q': {}}}, self._images()
      )

    self.assertEqual(stdout.getvalue(), '')

  def test_an_unreadable_card_is_not_evidence(self):
    self.mock_get.side_effect = requests.ConnectionError('no card route')
    self.mock_post.return_value = self._response(200, self._body())

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      result = infer.TypeSafeJevWrapper().predict_jev_mm(
          {'state': 's', 'questions': {'q': {}}}, self._images()
      )

    self.assertEqual(self.mock_post.call_count, 1)
    self.assertNotIn('warning', stdout.getvalue())
    self.assertIsNotNone(result[2])

  def test_a_non_object_card_is_not_evidence(self):
    # An OpenAI-compatible card route answers a bare array; the probe must
    # treat it as unreadable rather than raise.
    self.mock_get.return_value = self._response(
        200, json.dumps([{'id': 'kev-latest'}]).encode()
    )
    self.mock_post.return_value = self._response(200, self._body())

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      result = infer.TypeSafeJevWrapper().predict_jev_mm(
          {'state': 's', 'questions': {'q': {}}}, self._images()
      )

    self.assertEqual(stdout.getvalue(), '')
    self.assertIsNotNone(result[2])

  def test_the_hosted_key_never_rides_to_the_vision_endpoint(self):
    self.mock_post.return_value = self._response(200, self._body())
    self.mock_get.return_value = self._response(
        200, json.dumps({'models': [{'description': 'Clef head'}]}).encode()
    )
    wrapper = infer.TypeSafeJevWrapper()

    wrapper.predict_jev_mm({'state': 's', 'questions': {'q': {}}}, [])

    _, kwargs = self.mock_post.call_args
    self.assertNotIn('Authorization', kwargs['headers'])

  def test_the_vision_endpoint_has_its_own_key(self):
    with mock.patch.dict(os.environ, {'TYPESAFE_MM_API_KEY': 'vision-key'}):
      wrapper = infer.TypeSafeJevWrapper()
    self.mock_post.return_value = self._response(200, self._body())

    wrapper.predict_jev_mm({'state': 's', 'questions': {'q': {}}}, [])

    _, kwargs = self.mock_post.call_args
    self.assertEqual(
        kwargs['headers']['Authorization'], 'Bearer vision-key'
    )

  def test_an_open_local_server_is_asked_for_no_key(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      os.environ['TYPESAFE_MM_ENDPOINT'] = 'http://127.0.0.1:8008/v1/systemone'
      wrapper = infer.TypeSafeJevWrapper()
    self.mock_post.return_value = self._response(200, self._body())

    wrapper.predict_jev_mm({'state': 's', 'questions': {'q': {}}}, [])

    _, kwargs = self.mock_post.call_args
    self.assertNotIn('Authorization', kwargs['headers'])

  def test_a_budget_grows_a_small_frame_only_when_exact(self):
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    kept = Image.open(
        io.BytesIO(base64.b64decode(infer.array_to_png_b64(pixels, 4096)))
    )
    grown = Image.open(
        io.BytesIO(
            base64.b64decode(
                infer.array_to_png_b64(pixels, 4096, exact=True)
            )
        )
    )
    self.assertEqual(kept.size, (32, 32))
    self.assertEqual(grown.size, (56, 56))
if __name__ == "__main__":
  absltest.main()
