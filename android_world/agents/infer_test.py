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


if __name__ == "__main__":
  absltest.main()
