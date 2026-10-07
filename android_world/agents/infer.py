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

"""Some LLM inference interface."""

import abc
import base64
import dataclasses
import io
import json
import os
import time
from typing import Any, Optional
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import answer_types
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
from google.generativeai.types import safety_types
import numpy as np
from PIL import Image
import requests


ERROR_CALLING_LLM = 'Error calling LLM'


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
  """Converts a numpy array into a byte string for a JPEG image."""
  image = Image.fromarray(image)
  return image_to_jpeg_bytes(image)


# The pixel budgets the ac-jev-v2 rows were encoded under, from dohnuts:
# recipe.IMAGE_PIXELS for the frame being decided on and
# recipe.HISTORY_IMAGE_PIXELS for the history frames. The serving endpoint
# takes a bare list of base64 strings (kev.api.SystemOneRequest.images), so
# nothing tells the server which frame is which: the client owns the sizing,
# and JevImage.kind is how it knows which budget to apply.
JEV_PIXEL_PATCH = 28
JEV_CURRENT_PIXELS = 1024 ** 2
JEV_HISTORY_PIXELS = 512 ** 2


def _jev_pixel_budget(kind):
  """The pixel count the frame of this kind was trained under."""
  return JEV_HISTORY_PIXELS if kind == 'history' else JEV_CURRENT_PIXELS




def _resize_to_pixels(image, budget, exact):
  """Scales `image` to about `budget` pixels, on a 28-pixel lattice.

  The lattice is the vision patch factor, so the processor leaves the frame
  at the size chosen here instead of picking another one. `exact` also grows
  a frame smaller than the budget, which is what a history frame gets: the
  training collator rescaled every history image to its budget rather than
  only shrinking the ones over it.
  """
  width, height = image.size
  if width <= 0 or height <= 0:
    return image
  scale = (budget / (width * height)) ** 0.5
  if not exact and scale >= 1.0:
    return image

  def side(old):
    return max(JEV_PIXEL_PATCH,
               int(round(old * scale / JEV_PIXEL_PATCH)) * JEV_PIXEL_PATCH)

  return image.resize((side(width), side(height)))


def array_to_png_b64(image, budget=None, exact=False):
  """Base64-encodes an RGB array as a PNG, at about `budget` pixels.

  PNG rather than JPEG because the screenshots the model was trained on are
  lossless PNG files: a lossy re-encode blurs exactly the thin strokes the
  set-of-mark numbers are drawn with. A `budget` of None sends the frame at
  its native size.

  Args:
    image: An HxWx3 RGB array.
    budget: The pixel count to scale to, or None for the native size.
    exact: Whether a frame smaller than `budget` is grown to it.

  Returns:
    The base64 text of the PNG encoding of the (possibly resized) frame.
  """
  handle = Image.fromarray(image)
  if budget:
    handle = _resize_to_pixels(handle, budget, exact)
  in_mem_file = io.BytesIO()
  handle.save(in_mem_file, format='PNG')
  in_mem_file.seek(0)
  return base64.b64encode(in_mem_file.read()).decode('utf-8')


def jev_image_to_b64(image):
  """Encodes one JevImage under the pixel budget its kind names."""
  return array_to_png_b64(
      image.pixels,
      budget=_jev_pixel_budget(image.kind),
      exact=image.kind == 'history',
  )


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
  in_mem_file = io.BytesIO()
  image.save(in_mem_file, format='JPEG')
  # Reset file pointer to start
  in_mem_file.seek(0)
  img_bytes = in_mem_file.read()
  return img_bytes


class LlmWrapper(abc.ABC):
  """Abstract interface for (text only) LLM."""

  @abc.abstractmethod
  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Calling text-only LLM with a prompt.

    Args:
      text_prompt: Text prompt.

    Returns:
      Text output, is_safe, and raw output.
    """


class JevWrapper(abc.ABC):
  """Abstract interface for the Jev decision model."""

  @abc.abstractmethod
  def predict_jev(
      self,
      request: dict[str, Any],
  ) -> tuple[str, Optional[bool], Any]:
    """Calls Jev with a structured decision request.

    Args:
      request: The request body without the model field, i.e.
        {'state': ..., 'questions': ...}. The concrete wrapper adds its
        configured model name.

    Returns:
      Response JSON text, is_safe, and the parsed JSON response (a dict).
    """


class MultimodalLlmWrapper(abc.ABC):
  """Abstract interface for Multimodal LLM."""

  @abc.abstractmethod
  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """


@dataclasses.dataclass(frozen=True)
class JevImage:
  """One screenshot in a multimodal decision request.

  Attributes:
    kind: What the frame is to the model: 'history' for a past decision's
      screen, 'current' for the frame being decided on, 'marked' for the
      set-of-mark rendering of that same frame. The wire carries only a
      list of base64 strings, so the budget each kind is encoded under
      (512^2 for history, 1024^2 for the current frames, as the training
      recipe sets them) is applied here, before the pixels leave the client.
    pixels: The RGB image as an HxWx3 uint8 array.
  """

  kind: str
  pixels: np.ndarray


class MultimodalJevWrapper(abc.ABC):
  """Abstract interface for the multimodal Jev decision model."""

  @abc.abstractmethod
  def predict_jev_mm(
      self,
      request: dict[str, Any],
      images: list[JevImage],
  ) -> tuple[str, Optional[bool], Any]:
    """Calls the decision model with a structured request and its screenshots.

    Args:
      request: `{'state': ..., 'questions': ...}`, the same body the text-only
        wrapper sends.
      images: The frames in prompt order: history oldest first, then the raw
        current frame, then its marked rendering when the screen offers taps.

    Returns:
      Text output, is_safe, and raw output.
    """


SAFETY_SETTINGS_BLOCK_NONE = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
}


class GeminiGcpWrapper(LlmWrapper, MultimodalLlmWrapper):
  """Gemini GCP interface."""

  def __init__(
      self,
      model_name: str | None = None,
      max_retry: int = 3,
      temperature: float = 0.0,
      top_p: float = 0.95,
      enable_safety_checks: bool = True,
  ):
    if 'GCP_API_KEY' not in os.environ:
      raise RuntimeError('GCP API key not set.')
    genai.configure(api_key=os.environ['GCP_API_KEY'])
    self.llm = genai.GenerativeModel(
        model_name,
        safety_settings=None
        if enable_safety_checks
        else SAFETY_SETTINGS_BLOCK_NONE,
        generation_config=generation_types.GenerationConfig(
            temperature=temperature, top_p=top_p
        ),
    )
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)

  def predict(
      self,
      text_prompt: str,
      enable_safety_checks: bool = True,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(
        text_prompt, [], enable_safety_checks, generation_config
    )

  def is_safe(self, raw_response):
    try:
      return (
          raw_response.candidates[0].finish_reason
          != answer_types.FinishReason.SAFETY
      )
    except Exception:  # pylint: disable=broad-exception-caught
      #  Assume safe if the response is None or doesn't have candidates.
      return True

  def predict_mm(
      self,
      text_prompt: str,
      images: list[np.ndarray],
      enable_safety_checks: bool = True,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    counter = self.max_retry
    retry_delay = 1.0
    output = None
    while counter > 0:
      try:
        output = self.llm.generate_content(
            [text_prompt] + [Image.fromarray(image) for image in images],
            safety_settings=None
            if enable_safety_checks
            else SAFETY_SETTINGS_BLOCK_NONE,
            generation_config=generation_config,
        )
        return output.text, True, output
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2

    if (output is not None) and (not self.is_safe(output)):
      return ERROR_CALLING_LLM, False, output
    return ERROR_CALLING_LLM, None, None

  def generate(
      self,
      contents: (
          content_types.ContentsType | list[str | np.ndarray | Image.Image]
      ),
      safety_settings: safety_types.SafetySettingOptions | None = None,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Any]:
    """Exposes the generate_content API.

    Args:
      contents: The input to the LLM.
      safety_settings: Safety settings.
      generation_config: Generation config.

    Returns:
      The output text and the raw response.
    Raises:
      RuntimeError:
    """
    counter = self.max_retry
    retry_delay = 1.0
    response = None
    if isinstance(contents, list):
      contents = self.convert_content(contents)
    while counter > 0:
      try:
        response = self.llm.generate_content(
            contents=contents,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        return response.text, response
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2
    raise RuntimeError(f'Error calling LLM. {response}.')

  def convert_content(
      self,
      contents: list[str | np.ndarray | Image.Image],
  ) -> content_types.ContentsType:
    """Converts a list of contents to a ContentsType."""
    converted = []
    for item in contents:
      if isinstance(item, str):
        converted.append(item)
      elif isinstance(item, np.ndarray):
        converted.append(Image.fromarray(item))
      elif isinstance(item, Image.Image):
        converted.append(item)
    return converted


class Gpt4Wrapper(LlmWrapper, MultimodalLlmWrapper):
  """OpenAI GPT4 wrapper.

  Attributes:
    openai_api_key: The class gets the OpenAI api key either explicitly, or
      through env variable in which case just leave this empty.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: GPT model to use based on if it is multimodal.
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str,
      max_retry: int = 3,
      temperature: float = 0.0,
  ):
    if 'OPENAI_API_KEY' not in os.environ:
      raise RuntimeError('OpenAI API key not set.')
    self.openai_api_key = os.environ['OPENAI_API_KEY']
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    self.model = model_name

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.openai_api_key}',
    }

    payload = {
        'model': self.model,
        'temperature': self.temperature,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
            ],
        }],
        'max_tokens': 1000,
    }

    # Gpt-4v supports multiple images, just need to insert them in the content
    # list.
    for image in images:
      payload['messages'][0]['content'].append({
          'type': 'image_url',
          'image_url': {
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
          },
      })

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload,
        )
        if response.ok and 'choices' in response.json():
          return (
              response.json()['choices'][0]['message']['content'],
              None,
              response,
          )
        print(
            'Error calling OpenAI API with error message: '
            + response.json()['error']['message']
        )
        time.sleep(wait_seconds)
        wait_seconds *= 2
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
    return ERROR_CALLING_LLM, None, None


class OpenAIWrapper(LlmWrapper, MultimodalLlmWrapper):
  """OpenAI Compatible API wrapper.
  
  Attributes:
    openai_api_key: The class gets the OpenAI api key either explicitly, or
      through env variable in which case just leave this empty.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: GPT model to use based on if it is multimodal.
    base_url: The base URL for the OpenAI API.
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      base_url: str,
      model_name: str,
      max_retry: int = 3,
      temperature: float = 0.0,
  ):
    if 'OPENAI_API_KEY' not in os.environ:
      raise RuntimeError('OpenAI API key not set.')
    self.openai_api_key = os.environ['OPENAI_API_KEY']
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    self.model = model_name
    self.base_url = base_url
    try:
      from openai import OpenAI
      self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
    except ImportError:
      print(
          'OpenAI package not installed. Please install it to use OpenAIWrapper.'
      )
      self.client = None
    
  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.openai_api_key}',
    }

    payload = {
        'model': self.model,
        'temperature': self.temperature,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
            ],
        }],
        'max_tokens': 1000,
    }

    # Gpt-4v supports multiple images, just need to insert them in the content
    # list.
    for image in images:
      payload['messages'][0]['content'].append({
          'type': 'image_url',
          'image_url': {
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}'
          },
      })

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        if self.client is None:
          response = requests.post(
              f'{self.base_url}/v1/chat/completions',
              headers=headers,
              json=payload,
          )
          if response.ok and 'choices' in response.json():
            return (
                response.json()['choices'][0]['message']['content'],
                None,
                response,
            )
          print(
              'Error calling OpenAI API with error message: '
              + response.json()['error']['message']
          )
        else:
          response = self.client.chat.completions.create(
            model=self.model,
            messages=payload['messages'],
            temperature=self.temperature,
            max_tokens=1000,
            timeout=60
          )
          if response and response.choices:
            return (
                response.choices[0].message.content,
                None,
                response,
            )
          print(
              'Error calling OpenAI API with error message: '
              + str(response)
          )
        time.sleep(wait_seconds)
        wait_seconds *= 2
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
    return ERROR_CALLING_LLM, None, None    


class TypeSafeJevWrapper(JevWrapper, MultimodalJevWrapper):
  """TypeSafe systemone wrapper used by the mobile-jev policy.

  Sends one structured choice request per decision to the TypeSafe evaluation
  endpoint. Deliberately performs no retries: mobile-jev treats a transport
  failure as fatal for the run, and an action must never be replayed.

  `predict_jev` is the text-only call the v1 agent makes; `predict_jev_mm` adds
  the screenshots the ac-jev-v2 checkpoint is trained on. The two endpoints are
  separate settings because they are separate checkpoints: kev.serve answers
  both model names from the one checkpoint it loaded, and a checkpoint without a
  vision backbone drops the `images` field and answers from the state alone, so
  `predict_jev_mm` refuses to fall back to `endpoint` and asks, once, whether the
  vision endpoint really serves a vision checkpoint.

  Attributes:
    api_key: TYPESAFE_API_KEY. Empty is allowed when an endpoint was named
      explicitly, which is how a local kev.serve runs with KEV_API_KEY unset.
    mm_api_key: TYPESAFE_MM_API_KEY, the vision endpoint's own key (a kev.serve
      started with KEV_API_KEY set). Empty means that endpoint is asked for no
      key, and the text endpoint's key is never sent to it.
    model_name: The model to use; TYPESAFE_MODEL, or 'jev-latest'. kev.serve
      accepts that name beside its own 'kev-latest'.
    endpoint: The text-only decision endpoint; TYPESAFE_ENDPOINT overrides
      ENDPOINT.
    mm_endpoint: The multimodal decision endpoint from TYPESAFE_MM_ENDPOINT, or
      '' when no vision server is configured.
    timeout_sec: Per-request timeout in seconds.
  """

  ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
  # No default: a vision request must not silently reach the text-only
  # model, which would answer from the text alone and look like a working
  # run. Set TYPESAFE_MM_ENDPOINT (or MM_ENDPOINT here) to enable v2.
  MM_ENDPOINT = ''
  DEFAULT_MODEL = 'jev-latest'

  def __init__(
      self,
      model_name: str | None = None,
      timeout_sec: float = 30.0,
  ):
    endpoint = os.environ.get('TYPESAFE_ENDPOINT', '').strip()
    mm_endpoint = os.environ.get('TYPESAFE_MM_ENDPOINT', '').strip()
    api_key = os.environ.get('TYPESAFE_API_KEY', '').strip()
    if not api_key and not (endpoint or mm_endpoint):
      raise RuntimeError(
          'TypeSafe API key not set. Export TYPESAFE_API_KEY for the hosted'
          ' service, or point TYPESAFE_ENDPOINT / TYPESAFE_MM_ENDPOINT at a'
          ' local server that takes no key.'
      )
    self.api_key = api_key
    self.mm_api_key = os.environ.get('TYPESAFE_MM_API_KEY', '').strip()
    self.model_name = (
        model_name or os.environ.get('TYPESAFE_MODEL') or self.DEFAULT_MODEL
    )
    self.endpoint = endpoint or self.ENDPOINT
    self.mm_endpoint = mm_endpoint or self.MM_ENDPOINT
    self.timeout_sec = timeout_sec
    self._vision_checked = False

  def _headers(self, api_key: str | None = None) -> dict[str, str]:
    """The request headers; no Authorization for a server that wants none.

    The key is per endpoint: `api_key` names the text endpoint's (and is the
    default), `mm_api_key` the vision endpoint's, so the hosted credential
    never rides along to a different server.
    """
    headers = {'Content-Type': 'application/json'}
    key = self.api_key if api_key is None else api_key
    if key:
      headers['Authorization'] = f'Bearer {key}'
    return headers

  def _models_url(self) -> str:
    """The sibling /v1/models path of the endpoint, or '' when unrecognizable.

    kev.serve publishes the model card there, and this client only ever names
    the /v1/systemone path, so the card is one string replacement away.
    """
    for url in (self.mm_endpoint, self.endpoint):
      if url.endswith('/systemone'):
        return url[: -len('/systemone')] + '/models'
    return ''

  def _warn_if_text_only(self) -> None:
    """Asks the vision endpoint, once, whether it can see at all.

    A kev.serve that loaded a text-only checkpoint answers a multimodal request
    happily and without the frames: the run looks healthy while the model
    decides from the state alone. The card names the head it loaded, so a card
    that mentions neither a clef nor a vision backbone is said out loud.
    Nothing is raised: an unreadable card is not evidence, and a wrong guess
    about the server must not end a run that may be fine.
    """
    if self._vision_checked:
      return
    self._vision_checked = True
    url = self._models_url()
    if not url:
      return
    try:
      response = requests.get(
          url, headers=self._headers(self.mm_api_key),
          timeout=self.timeout_sec
      )
      if not response.ok:
        return
      body = response.json()
    except (requests.RequestException, ValueError):
      return
    if not isinstance(body, dict):
      # A card route that answers a bare array (the OpenAI-compatible shape)
      # is as unreadable as one that answers nothing; it must not raise.
      return
    models = body.get('models') or []
    described = ' '.join(
        str(model.get('description', ''))
        for model in models
        if isinstance(model, dict)
    )
    if described and 'lef' not in described and 'ision' not in described:
      print(
          'Jev warning: ' + self.mm_endpoint + ' reports ' + described[:120]
          + ', which does not look like a vision checkpoint; that endpoint'
          ' answers from the state and drops the images this agent sends.'
      )

  def _redact(self, text: str) -> str:
    """Redacts the API keys from an error message and bounds its length."""
    for key in (self.api_key, self.mm_api_key):
      if key:
        text = text.replace(key, '[redacted]')
    return text[:500]

  def predict_jev(
      self,
      request: dict[str, Any],
  ) -> tuple[str, Optional[bool], Any]:
    """Sends one decision request and returns the raw response.

    Args:
      request: {'state': ..., 'questions': ...}.

    Returns:
      (response JSON text, None, parsed response dict) on success, or
      (ERROR_CALLING_LLM, False, None) on any failure. Failures are never
      retried.
    """
    payload = {'model': self.model_name, **request}
    try:
      response = requests.post(
          self.endpoint,
          headers=self._headers(),
          json=payload,
          timeout=self.timeout_sec,
      )
    except requests.RequestException as e:
      print(f'Error calling Jev: {self._redact(str(e))}')
      return ERROR_CALLING_LLM, False, None
    if not response.ok:
      print(
          f'Jev returned HTTP {response.status_code}: '
          f'{self._redact(response.text)}'
      )
      return ERROR_CALLING_LLM, False, None
    try:
      parsed = response.json()
    except ValueError:
      print('Jev returned invalid JSON.')
      return ERROR_CALLING_LLM, False, None
    if not isinstance(parsed, dict) or not isinstance(
        parsed.get('answers'), dict
    ):
      print('Jev response is missing answers.')
      return ERROR_CALLING_LLM, False, None
    return json.dumps(parsed, ensure_ascii=False), None, parsed


  def predict_jev_mm(
      self,
      request: dict[str, Any],
      images: list[JevImage],
  ) -> tuple[str, Optional[bool], Any]:
    """Sends one multimodal decision request and returns the raw response.

    The body is the text-only body plus an `images` list of base64 PNGs,
    in the order the prompt reads them: one placeholder per frame, history
    oldest first, then the raw frame, then its marked rendering. Nothing in
    that list says which frame is which, so each one is encoded under the
    pixel budget its kind was trained on before it goes out.

    Args:
      request: `{'state': ..., 'questions': ...}`.
      images: The frames, in prompt order.

    Returns:
      (response JSON text, None, parsed response dict) on success, or
      (ERROR_CALLING_LLM, False, None) on any failure. Failures are never
      retried, because a retry could replay an action.
    """
    if not self.mm_endpoint:
      raise RuntimeError(
          'No multimodal decision endpoint is configured; set'
          ' TYPESAFE_MM_ENDPOINT to the server that renders the ac-jev-v2'
          ' prompt with its screenshots.'
      )
    self._warn_if_text_only()
    payload = {
        'model': self.model_name,
        **request,
        # kev.api.SystemOneRequest.images is a bare list of base64 strings, so
        # the server learns the order of the frames and nothing else: the pixel
        # budget each kind was trained under is spent here, on the way out.
        'images': [jev_image_to_b64(image) for image in images],
    }
    try:
      response = requests.post(
          self.mm_endpoint,
          headers=self._headers(self.mm_api_key),
          json=payload,
          timeout=self.timeout_sec,
      )
    except requests.RequestException as e:
      print(f'Error calling Jev: {self._redact(str(e))}')
      return ERROR_CALLING_LLM, False, None
    if not response.ok:
      print(
          f'Jev returned HTTP {response.status_code}: '
          f'{self._redact(response.text)}'
      )
      return ERROR_CALLING_LLM, False, None
    try:
      parsed = response.json()
    except ValueError:
      print('Jev returned invalid JSON.')
      return ERROR_CALLING_LLM, False, None
    if not isinstance(parsed, dict) or not isinstance(
        parsed.get('answers'), dict
    ):
      print('Jev response is missing answers.')
      return ERROR_CALLING_LLM, False, None
    return json.dumps(parsed, ensure_ascii=False), None, parsed
