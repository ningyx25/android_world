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

"""Offline tests for the ClientMobileJev agent and its policy."""

import json
import pickle
import time
from unittest import mock
from absl.testing import absltest
from android_world.agents import infer
from android_world.agents import mobile_jev
from android_world.env import interface
from android_world.env import representation_utils
import numpy as np


SCREEN = (1000, 2000)


def _ui_element(
    text='',
    label='',
    class_name='android.widget.TextView',
    bounds=(0, 0, 100, 50),
    clickable=False,
    editable=False,
    scrollable=False,
    enabled=True,
    focused=False,
    resource_id=None,
    hint=None,
    visible=True,
):
  """Builds a UIElement through the real serialization round-trip."""
  left, top, right, bottom = bounds
  box = representation_utils.BoundingBox(left, right, top, bottom)
  return representation_utils.ui_element_from_dict(
      representation_utils.ui_element_to_dict(
          representation_utils.UIElement(
              text=text or None,
              content_description=label or None,
              class_name=class_name,
              bbox=box,
              bbox_pixels=box,
              hint_text=hint,
              resource_id=resource_id,
              is_clickable=True if clickable else None,
              is_editable=True if editable else None,
              is_scrollable=True if scrollable else None,
              is_enabled=enabled,
              is_focused=True if focused else None,
              is_visible=visible,
          )
      )
  )


def _observation(
    elements,
    package='com.example',
    screen=SCREEN,
    observed_at=None,
):
  state = interface.State(
      pixels=np.zeros((10, 10, 3), dtype=np.uint8),
      forest=None,
      ui_elements=list(elements),
      auxiliaries={},
  )
  return mobile_jev.summarize_state(
      state, package, screen, observed_at=observed_at
  )


def _answer(criteria, choice=None, confidence=1.0):
  keys = list(criteria)
  chosen = choice if choice is not None else keys[0]
  probabilities = {key: (1.0 if key == chosen else 0.0) for key in keys}
  return {
      'type': 'choice',
      'choice': chosen,
      'probabilities': probabilities,
      'confidence': confidence,
  }


def _response(answers):
  return {
      'model': 'jev-test',
      'answers': answers,
      'usage': {'input_tokens': 1, 'output_tokens': 1},
  }


def _operation_response(operation='TAP', target=None):
  """Builds a responder that answers with one operation/target."""
  heads = {
      'OPEN_APP': 'app_target',
      'TAP': 'tap_target',
      'TYPE_TEXT': 'text_value',
      'SCROLL_DOWN': 'scroll_target',
      'SCROLL_UP': 'scroll_target',
      'SCROLL_LEFT': 'scroll_target',
      'SCROLL_RIGHT': 'scroll_target',
  }

  def responder(request):
    questions = request['questions']
    answers = {
        'operation': _answer(questions['operation']['criteria'], operation)
    }
    head = heads.get(operation)
    if target is not None and head in questions:
      answers[head] = _answer(questions[head]['criteria'], target)
    return _response(answers)

  return responder


class FakeJevWrapper(infer.JevWrapper):
  """Scripted Jev wrapper that never touches the network."""

  def __init__(self, responder):
    self.model_name = 'jev-test'
    self.responder = responder
    self.requests = []

  def predict_jev(self, request):
    self.requests.append(request)
    if callable(self.responder):
      response = self.responder(request)
    else:
      response = self.responder.pop(0)
    if response is None:
      return infer.ERROR_CALLING_LLM, False, None
    return json.dumps(response), None, response


class FakeAndroidEnvClient:
  """Duck-typed stand-in for interface.AndroidEnvClient."""

  def __init__(
      self,
      ui_elements=None,
      activity='com.example/.Main',
      apps=(),
      screen_size=SCREEN,
  ):
    self.ui_elements = list(ui_elements or [])
    self.activity = activity
    self.apps = list(apps)
    self.screen_size = screen_size
    self.state_queue = []
    self.screen_size_queue = []
    self.reset_go_home = None
    self.hide_automation_ui_called = False
    self.executed_actions = []
    self.taps = []
    self.generated_swipes = []
    self.typed_texts = []
    self.pressed_keys = []
    self.generic_requests = []
    self.get_state_calls = 0
    self.fail_get_state_after = None
    self.get_all_apps_failures = 0

  def reset(self, go_home: bool) -> None:
    self.reset_go_home = go_home

  def hide_automation_ui(self) -> None:
    self.hide_automation_ui_called = True

  def queue_state(self, ui_elements) -> None:
    self.state_queue.append(list(ui_elements))

  def get_state(self, wait_to_stabilize: bool = False) -> interface.State:
    del wait_to_stabilize
    self.get_state_calls += 1
    if (
        self.fail_get_state_after is not None
        and self.get_state_calls > self.fail_get_state_after
    ):
      raise RuntimeError('state read failed')
    if self.state_queue:
      ui_elements = self.state_queue.pop(0)
    else:
      ui_elements = self.ui_elements
    return interface.State(
        pixels=np.zeros((10, 10, 3), dtype=np.uint8),
        forest=None,
        ui_elements=list(ui_elements),
        auxiliaries={},
    )

  def get_current_activity(self, timeout_sec: float = 10) -> str:
    del timeout_sec
    return self.activity

  def get_all_apps(self, timeout_sec: float = 10):
    del timeout_sec
    if self.get_all_apps_failures > 0:
      self.get_all_apps_failures -= 1
      raise RuntimeError('installed app list unavailable')
    return list(self.apps)

  def get_logical_screen_size(self) -> tuple[int, int]:
    if self.screen_size_queue:
      return self.screen_size_queue.pop(0)
    return self.screen_size

  def get_physical_frame_boundary(self) -> tuple[int, int, int, int]:
    return (0, 0) + self.screen_size

  def get_orientation(self) -> int:
    return 0

  def execute_action(self, action) -> None:
    self.executed_actions.append(action)

  def tap(self, x: int, y: int, timeout_sec: float = 10) -> None:
    del timeout_sec
    self.taps.append((x, y))

  def generate_swipe_command(
      self,
      start_x: int,
      start_y: int,
      end_x: int,
      end_y: int,
      duration_ms: int | None = None,
  ) -> list[str]:
    self.generated_swipes.append(
        (start_x, start_y, end_x, end_y, duration_ms)
    )
    return [
        'shell', 'input', 'swipe', str(start_x), str(start_y),
        str(end_x), str(end_y), str(duration_ms or 1000),
    ]

  def issue_generic_request(self, args, timeout_sec: float = 10):
    del timeout_sec
    self.generic_requests.append(args)
    return {}

  def press_key(self, keycode: str, timeout_sec: float = 10) -> None:
    del timeout_sec
    self.pressed_keys.append(keycode)

  def type_text(self, text: str, timeout_sec: float = 10) -> None:
    del timeout_sec
    self.typed_texts.append(text)


def _create_agent(
    client=None,
    responder=None,
    confidence_threshold=0.0,
    wait_timeout_ms=mobile_jev.WAIT_TIMEOUT_MS,
    input_timeout_ms=mobile_jev.INPUT_TIMEOUT_MS,
):
  client = client or FakeAndroidEnvClient()
  jev = FakeJevWrapper(responder)
  agent = mobile_jev.ClientMobileJev(
      client,
      llm=None,
      jev=jev,
      confidence_threshold=confidence_threshold,
      wait_timeout_ms=wait_timeout_ms,
      input_timeout_ms=input_timeout_ms,
  )
  agent.transition_pause = 0.0
  return agent, client, jev


class TextCandidatesTest(absltest.TestCase):

  def test_exact_spans_and_internal_whitespace(self):
    candidates = mobile_jev.text_candidates('Type New York now')
    self.assertEqual(candidates.source, 'goal')
    self.assertIn('New York', candidates.values)
    self.assertIn('Type New York', candidates.values)
    self.assertIn('York now', candidates.values)

  def test_strips_surrounding_quotes_and_punctuation(self):
    candidates = mobile_jev.text_candidates('Type "New York".')
    self.assertIn('New York', candidates.values)

  def test_supplied_values_win(self):
    candidates = mobile_jev.text_candidates('Type anything', ['a', 'b', 'a'])
    self.assertEqual(candidates.source, 'supplied')
    self.assertEqual(candidates.values, ('a', 'b'))

  def test_overflow_offers_no_text(self):
    goal = ' '.join(f'w{i}' for i in range(300))
    candidates = mobile_jev.text_candidates(goal)
    self.assertTrue(candidates.overflow)
    self.assertEmpty(candidates.values)


class CandidatesTest(absltest.TestCase):

  def test_taps_require_actionable_and_enabled(self):
    observation = _observation([
        _ui_element(text='Clickable', clickable=True),
        _ui_element(text='Editable', editable=True),
        _ui_element(text='Plain text'),
        _ui_element(text='Disabled', clickable=True, enabled=False),
    ])
    actions = mobile_jev.candidates_for(observation)
    self.assertIn('tap_0', actions)
    self.assertIn('tap_1', actions)
    self.assertNotIn('tap_2', actions)
    self.assertNotIn('tap_3', actions)
    self.assertIn('back', actions)
    self.assertIn('home', actions)

  def test_scroll_candidates_skip_nested_and_duplicate_regions(self):
    observation = _observation([
        _ui_element(text='Outer', scrollable=True, bounds=(0, 0, 1000, 1000)),
        _ui_element(text='Inner', scrollable=True, bounds=(0, 0, 900, 900)),
        _ui_element(text='Twin', scrollable=True, bounds=(0, 0, 900, 900)),
        _ui_element(
            text='Small', scrollable=True, bounds=(500, 1200, 900, 1500)
        ),
    ])
    actions = mobile_jev.candidates_for(observation)
    # Outer is skipped only if another scrollable region is >= 70% of its area
    # and is not the same bounds; the Inner/Twin pair keeps its first entry.
    self.assertNotIn('scroll_down_0', actions)
    self.assertNotIn('scroll_down_2', actions)
    self.assertIn('scroll_down_1', actions)
    self.assertIn('scroll_down_3', actions)

  def test_enter_and_text_only_when_editable(self):
    editable = _observation([
        _ui_element(text='', editable=True, focused=True,
                    class_name='android.widget.EditText'),
    ])
    self.assertIn('enter', mobile_jev.candidates_for(editable, ['Tokyo']))
    self.assertIn('text_0', mobile_jev.candidates_for(editable, ['Tokyo']))
    plain = _observation([_ui_element(text='Plain', clickable=True)])
    self.assertNotIn('enter', mobile_jev.candidates_for(plain, ['Tokyo']))


class ValidateChoiceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.criteria = {'A': 'a', 'B': 'b'}

  def test_accepts_valid_distribution(self):
    answer = {
        'type': 'choice',
        'choice': 'A',
        'probabilities': {'A': 0.7, 'B': 0.3},
        'confidence': 0.7,
    }
    self.assertIs(mobile_jev.validate_choice(answer, self.criteria), answer)

  def test_rejects_malformed_distributions(self):
    valid = {
        'type': 'choice',
        'choice': 'A',
        'probabilities': {'A': 0.7, 'B': 0.3},
        'confidence': 0.7,
    }
    cases = {
        'missing choice': {**valid, 'choice': 'C'},
        'wrong type': {**valid, 'type': 'noul'},
        'missing key': {**valid, 'probabilities': {'A': 1.0}},
        'extra key': {
            **valid,
            'probabilities': {'A': 0.5, 'B': 0.25, 'C': 0.25},
        },
        'bad sum': {
            **valid,
            'probabilities': {'A': 0.5, 'B': 0.2},
        },
        'non argmax': {
            **valid,
            'probabilities': {'A': 0.3, 'B': 0.7},
        },
        'nan confidence': {**valid, 'confidence': float('nan')},
        'infinity': {
            **valid,
            'probabilities': {'A': float('inf'), 'B': 0.0},
        },
        'list probabilities': {**valid, 'probabilities': [0.7, 0.3]},
        'bool probability': {
            **valid,
            'probabilities': {'A': True, 'B': 0.0},
        },
    }
    for name, answer in cases.items():
      with self.subTest(name=name):
        with self.assertRaises(mobile_jev.InvalidChoiceError):
          mobile_jev.validate_choice(answer, self.criteria)


class BuildQuestionsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.observation = _observation([
        _ui_element(text='Wi-Fi', clickable=True, bounds=(0, 0, 200, 50)),
        _ui_element(
            text='',
            editable=True,
            focused=True,
            class_name='android.widget.EditText',
            bounds=(0, 100, 200, 150),
        ),
        _ui_element(text='List', scrollable=True, bounds=(0, 200, 1000, 1000)),
    ])
    self.space = mobile_jev.build_questions(
        self.observation, ['London', 'Berlin'], ['Notes']
    )

  def test_question_ids_and_operation_order(self):
    self.assertEqual(
        list(self.space.questions),
        ['operation', 'app_target', 'tap_target', 'scroll_target',
         'text_value'],
    )
    self.assertEqual(
        list(self.space.questions['operation']['criteria']),
        ['OPEN_APP', 'TAP', 'TYPE_TEXT', 'SCROLL_DOWN', 'SCROLL_UP',
         'SCROLL_LEFT', 'SCROLL_RIGHT', 'BACK', 'HOME', 'ENTER', 'WAIT',
         'DONE', 'BLOCKED'],
    )

  def test_target_criteria(self):
    self.assertEqual(
        self.space.questions['tap_target']['criteria'],
        {'1': '[1] Wi-Fi', '2': '[2] Focus text input: empty input field'},
    )
    self.assertEqual(
        self.space.questions['scroll_target']['criteria'],
        {'3': 'Scrollable region [3] List'},
    )
    self.assertEqual(
        self.space.questions['text_value']['criteria'],
        {'1': 'London', '2': 'Berlin',
         'NONE': self.space.questions['text_value']['criteria']['NONE']},
    )

  def test_element_entry_operations(self):
    entries = {entry['index']: entry for entry in self.space.elements}
    self.assertEqual(entries['1']['operations'], ['TAP'])
    self.assertEqual(entries['2']['operations'], ['TAP'])
    self.assertEqual(
        entries['3']['operations'],
        ['SCROLL_DOWN', 'SCROLL_UP', 'SCROLL_RIGHT', 'SCROLL_LEFT'],
    )


class PolicyDecideTest(absltest.TestCase):

  def _policy(self, responder, threshold=0.0):
    return mobile_jev.MobileJevPolicy(
        FakeJevWrapper(responder), threshold=threshold
    )

  def test_tap_decision_consumes_only_the_selected_branch(self):
    observation = _observation([
        _ui_element(text='Wi-Fi', clickable=True, bounds=(0, 0, 200, 50)),
        _ui_element(text='List', scrollable=True, bounds=(0, 100, 200, 300)),
    ])

    def responder(request):
      questions = request['questions']
      return _response({
          'operation': _answer(questions['operation']['criteria'], 'TAP'),
          'tap_target': _answer(questions['tap_target']['criteria'], '1'),
          'scroll_target': {'malicious': 'unused'},
      })

    decision = self._policy(responder).decide('Tap Wi-Fi', observation)
    self.assertEqual(decision.status, mobile_jev.Status.ACTION)
    self.assertEqual(decision.action.kind, mobile_jev.ActionKind.TAP)
    self.assertEqual(decision.action.element_id, '0')
    self.assertEqual(decision.choice, 'tap_0')
    self.assertIsNone(decision.reason)

  def test_request_is_text_only(self):
    observation = _observation([_ui_element(text='Wi-Fi', clickable=True)])
    wrapper = FakeJevWrapper(_operation_response('DONE'))
    decision = mobile_jev.MobileJevPolicy(wrapper).decide(
        'Do nothing', observation
    )
    self.assertEqual(decision.status, mobile_jev.Status.DONE)
    serialized = json.dumps(wrapper.requests[0])
    self.assertNotIn('image', serialized)
    self.assertNotIn('base64', serialized)
    self.assertEqual(
        set(wrapper.requests[0]['state']),
        {'goal', 'app', 'isEditable', 'textSource',
         'textEntryAvailableAfterFocus', 'visibleText', 'elements',
         'availableApps', 'recentActions'},
    )

  def test_named_apps_narrow_the_inventory(self):
    observation = _observation([])
    wrapper = FakeJevWrapper(_operation_response('DONE'))
    mobile_jev.MobileJevPolicy(wrapper).decide(
        'Open Notes', observation, apps=['Notes', 'Calculator', 'Clock']
    )
    app_criteria = wrapper.requests[0]['questions']['app_target']['criteria']
    self.assertEqual(app_criteria, {'1': 'Notes'})
    self.assertEqual(
        wrapper.requests[0]['state']['availableApps'],
        [{'index': '1', 'label': 'Notes'}],
    )

  def test_apps_are_capped(self):
    observation = _observation([])
    wrapper = FakeJevWrapper(_operation_response('DONE'))
    mobile_jev.MobileJevPolicy(wrapper).decide(
        'Open something', observation, apps=[f'App{i}' for i in range(250)]
    )
    self.assertLen(wrapper.requests[0]['state']['availableApps'], 200)

  def test_needs_input_when_no_text_span_fits(self):
    observation = _observation([
        _ui_element(text='', editable=True, focused=True,
                    class_name='android.widget.EditText'),
    ])

    def responder(request):
      questions = request['questions']
      return _response({
          'operation': _answer(
              questions['operation']['criteria'], 'TYPE_TEXT'
          ),
          'text_value': _answer(
              questions['text_value']['criteria'], 'NONE'
          ),
      })

    decision = self._policy(responder).decide('Type Tokyo', observation)
    self.assertEqual(decision.status, mobile_jev.Status.NEEDS_INPUT)
    self.assertIsNone(decision.action)
    self.assertIn('field value', decision.reason)

  def test_uncertain_status_uses_the_threshold(self):
    observation = _observation([_ui_element(text='Wi-Fi', clickable=True)])

    def responder(request):
      questions = request['questions']
      return _response({
          'operation': _answer(
              questions['operation']['criteria'], 'TAP', confidence=0.4
          ),
          'tap_target': _answer(
              questions['tap_target']['criteria'], '1', confidence=0.4
          ),
      })

    decision = self._policy(responder, threshold=0.9).decide(
        'Tap Wi-Fi', observation
    )
    self.assertEqual(decision.status, mobile_jev.Status.UNCERTAIN)
    self.assertEqual(
        self._policy(responder, threshold=0.0).decide(
            'Tap Wi-Fi', observation
        ).status,
        mobile_jev.Status.ACTION,
    )

  def test_focused_field_is_sent(self):
    observation = _observation([
        _ui_element(text='', editable=True, focused=True,
                    class_name='android.widget.EditText',
                    resource_id='com.example:id/input'),
    ])
    wrapper = FakeJevWrapper(
        _operation_response('TYPE_TEXT', target='1')
    )
    mobile_jev.MobileJevPolicy(wrapper).decide('Type London', observation)
    focused = wrapper.requests[0]['state']['focusedField']
    self.assertTrue(focused['editable'])
    self.assertEqual(focused['index'], '1')

  def test_oversized_payload_raises(self):
    observation = _observation([
        _ui_element(text='x' * 160_000, clickable=True),
    ])
    wrapper = FakeJevWrapper(_operation_response('DONE'))
    with self.assertRaises(mobile_jev.PayloadTooLargeError):
      mobile_jev.MobileJevPolicy(wrapper).decide('Huge', observation)
    self.assertEmpty(wrapper.requests)

  def test_model_failure_raises(self):
    observation = _observation([_ui_element(text='Wi-Fi', clickable=True)])
    with self.assertRaisesRegex(RuntimeError, 'Error calling Jev'):
      self._policy(lambda request: None).decide('Tap Wi-Fi', observation)


class FreshnessTest(absltest.TestCase):

  def test_bounds_change_is_tolerated_for_taps(self):
    before = _observation([
        _ui_element(text='Wi-Fi', clickable=True, bounds=(0, 0, 100, 50)),
    ])
    after = _observation([
        _ui_element(text='Wi-Fi', clickable=True, bounds=(0, 10, 100, 60)),
    ])
    mobile_jev.assert_fresh(
        after, before,
        mobile_jev._ActionSpec(mobile_jev.ActionKind.TAP, element_id='0'),
    )

  def test_changed_meaning_is_stale(self):
    before = _observation([
        _ui_element(text='Wi-Fi', clickable=True, bounds=(0, 0, 100, 50)),
    ])
    after = _observation([
        _ui_element(text='Wi-Fi on', clickable=True, bounds=(0, 0, 100, 50)),
    ])
    with self.assertRaises(mobile_jev.StaleObservationError):
      mobile_jev.assert_fresh(
          after, before,
          mobile_jev._ActionSpec(mobile_jev.ActionKind.TAP, element_id='0'),
      )

  def test_expired_observation_is_stale(self):
    observation = _observation(
        [_ui_element(text='Wi-Fi', clickable=True)],
        observed_at=time.monotonic() - 31,
    )
    with self.assertRaises(mobile_jev.StaleObservationError):
      mobile_jev.assert_fresh(
          observation, observation,
          mobile_jev._ActionSpec(mobile_jev.ActionKind.TAP, element_id='0'),
      )

  def test_package_change_is_stale(self):
    before = _observation([_ui_element(text='Wi-Fi', clickable=True)])
    after = _observation(
        [_ui_element(text='Wi-Fi', clickable=True)], package='com.other'
    )
    with self.assertRaises(mobile_jev.StaleObservationError):
      mobile_jev.assert_fresh(
          after, before,
          mobile_jev._ActionSpec(mobile_jev.ActionKind.TAP, element_id='0'),
      )

  def test_home_is_always_fresh_after_the_global_gate(self):
    before = _observation([_ui_element(text='Wi-Fi', clickable=True)])
    after = _observation([_ui_element(text='Changed', clickable=True)])
    mobile_jev.assert_fresh(
        after, before,
        mobile_jev._ActionSpec(mobile_jev.ActionKind.GLOBAL, name='home'),
    )

  def test_back_ignores_plain_text_but_not_controls(self):
    before = _observation([
        _ui_element(text='Clock 12:00'),
        _ui_element(text='Wi-Fi', clickable=True),
    ])
    clock_changed = _observation([
        _ui_element(text='Clock 12:01'),
        _ui_element(text='Wi-Fi', clickable=True),
    ])
    mobile_jev.assert_fresh(
        clock_changed, before,
        mobile_jev._ActionSpec(mobile_jev.ActionKind.GLOBAL, name='back'),
    )
    control_changed = _observation([
        _ui_element(text='Clock 12:00'),
        _ui_element(text='Bluetooth', clickable=True),
    ])
    with self.assertRaises(mobile_jev.StaleObservationError):
      mobile_jev.assert_fresh(
          control_changed, before,
          mobile_jev._ActionSpec(mobile_jev.ActionKind.GLOBAL, name='back'),
      )

  def test_type_checks_the_focused_input(self):
    def elements(text):
      return [_ui_element(
          text=text,
          editable=True,
          focused=True,
          class_name='android.widget.EditText',
          resource_id='com.example:id/input',
      )]

    before = _observation(elements(''))
    action = mobile_jev._ActionSpec(
        mobile_jev.ActionKind.TYPE, text='Tokyo', element_id='0'
    )
    mobile_jev.assert_fresh(_observation(elements('')), before, action)
    with self.assertRaises(mobile_jev.StaleObservationError):
      mobile_jev.assert_fresh(_observation(elements('old')), before, action)
    with self.assertRaises(mobile_jev.StaleObservationError):
      mobile_jev.assert_fresh(
          _observation(elements(''), package='com.other'), before, action
      )


class ActionMappingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.elements = [
        _ui_element(text='Wi-Fi', clickable=True, bounds=(0, 0, 200, 50)),
        _ui_element(text='List', scrollable=True, bounds=(0, 100, 200, 400)),
        _ui_element(
            text='',
            editable=True,
            focused=True,
            class_name='android.widget.EditText',
            bounds=(0, 500, 200, 550),
        ),
    ]

  def _agent(self, responder, elements=None, apps=()):
    client = FakeAndroidEnvClient(
        ui_elements=self.elements if elements is None else elements,
        apps=apps,
    )
    return _create_agent(client, responder)

  def test_tap_uses_the_verified_element_center(self):
    agent, client, _ = self._agent(_operation_response('TAP', '1'))
    agent.step('Tap Wi-Fi')
    # Element 0 has bounds (0, 0, 200, 50); dispatch is a coordinate tap so the
    # server never has to resolve an index against its own, possibly empty,
    # element list.
    self.assertEqual(client.taps, [(100, 25)])
    self.assertEmpty(client.executed_actions)

  def test_scroll_swipes_inside_the_verified_region(self):
    agent, client, _ = self._agent(_operation_response('SCROLL_DOWN', '3'))
    agent.step('Scroll the list')
    # Region bounds (0, 100, 200, 400): x=100, y1=160, y2=340; scrolling down
    # moves the finger from 80% to 20% of the region height.
    self.assertEqual(client.generated_swipes, [(100, 340, 100, 160, 300)])
    self.assertEqual(client.generic_requests[-1], [
        'shell', 'input', 'swipe', '100', '340', '100', '160', '300',
    ])
    self.assertEmpty(client.executed_actions)

  def test_enter_back_and_home(self):
    for operation, expected in (
        ('ENTER', 'keyboard_enter'),
        ('BACK', 'navigate_back'),
        ('HOME', 'navigate_home'),
    ):
      with self.subTest(operation=operation):
        agent, client, _ = self._agent(_operation_response(operation))
        agent.step('Press something')
        self.assertEqual(client.executed_actions[0].action_type, expected)

  def test_open_app_uses_the_offered_label(self):
    agent, client, _ = self._agent(
        _operation_response('OPEN_APP', '1'), apps=['Notes']
    )
    agent.step('Open Notes')
    action = client.executed_actions[0]
    self.assertEqual(action.action_type, 'open_app')
    self.assertEqual(action.app_name, 'Notes')

  def test_type_focuses_clears_and_types_without_submitting(self):
    elements = [_ui_element(
        text='',
        editable=True,
        focused=True,
        class_name='android.widget.EditText',
        resource_id='com.example:id/input',
        bounds=(0, 0, 500, 100),
    )]

    def responder(request):
      questions = request['questions']
      criteria = questions['text_value']['criteria']
      key = next(k for k, v in criteria.items() if v == 'Tokyo')
      return _response({
          'operation': _answer(
              questions['operation']['criteria'], 'TYPE_TEXT'
          ),
          'text_value': _answer(criteria, key),
      })

    agent, client, _ = self._agent(responder, elements=elements)
    # Reads: initial observe, pre-dispatch, after-action.
    client.queue_state(elements)
    client.queue_state(elements)
    client.queue_state([_ui_element(
        text='Tokyo',
        editable=True,
        focused=True,
        class_name='android.widget.EditText',
        resource_id='com.example:id/input',
        bounds=(0, 0, 500, 100),
    )])
    with mock.patch.object(mobile_jev.time, 'sleep'):
      step_data = agent.step('Type Tokyo')
    self.assertEqual(client.typed_texts, ['Tokyo'])
    self.assertEqual(client.pressed_keys, ['KEYCODE_DEL'])
    self.assertEqual(
        client.generic_requests,
        [['shell', 'input', 'keycombination', '113', '29']],
    )
    # Focusing the field is a coordinate tap; the server's INPUT_TEXT action is
    # never used because it presses ENTER.
    self.assertEqual(client.taps, [(250, 50)])
    self.assertEmpty(client.executed_actions)
    self.assertFalse(step_data.done)


class CoordinateDispatchTest(absltest.TestCase):
  """Tests the coordinate helpers used for dispatch."""

  def test_scroll_gestures_for_every_direction(self):
    observation = _observation([
        _ui_element(text='List', scrollable=True, bounds=(0, 100, 200, 400)),
    ])
    # Region insets: x=100, y=250 (center) and x=40/160, y=160/340 (20%/80%).
    expected = {
        'down': (100, 340, 100, 160),
        'up': (100, 160, 100, 340),
        'right': (160, 250, 40, 250),
        'left': (40, 250, 160, 250),
    }
    for direction, gesture in expected.items():
      with self.subTest(direction=direction):
        self.assertEqual(
            mobile_jev._scroll_gesture(observation, '0', direction), gesture
        )

  def test_missing_targets_raise(self):
    observation = _observation([])
    with self.assertRaisesRegex(
        RuntimeError, 'missing from the observation'
    ):
      mobile_jev._element_center(observation, '0')
    with self.assertRaisesRegex(RuntimeError, 'scroll region'):
      mobile_jev._scroll_gesture(observation, '0', 'down')


class ClientMobileJevStepTest(absltest.TestCase):

  def _agent(self, responder, **kwargs):
    return _create_agent(
        FakeAndroidEnvClient(
            ui_elements=[_ui_element(text='Wi-Fi', clickable=True)],
            apps=['Notes'],
        ),
        responder,
        **kwargs,
    )

  def test_done_ends_without_dispatching(self):
    agent, client, _ = self._agent(_operation_response('DONE'))
    result = agent.step('Do nothing')
    self.assertTrue(result.done)
    self.assertEmpty(client.executed_actions)
    self.assertEmpty(client.taps)
    self.assertEqual(result.data['mobile_jev']['status'], 'done')

  def test_blocked_ends_without_dispatching(self):
    agent, client, _ = self._agent(_operation_response('BLOCKED'))
    result = agent.step('Impossible')
    self.assertTrue(result.done)
    self.assertEmpty(client.executed_actions)
    self.assertEmpty(client.taps)
    self.assertEqual(result.data['mobile_jev']['status'], 'blocked')

  def test_agent_has_no_env_attribute(self):
    agent, _, _ = self._agent(_operation_response('DONE'))
    self.assertFalse(hasattr(agent, 'env'))

  def test_reset_clears_state(self):
    agent, client, _ = self._agent(_operation_response('DONE'))
    agent.step('Do nothing')
    self.assertLen(agent.history, 1)
    agent.reset(go_home_on_reset=True)
    self.assertEmpty(agent.history)
    self.assertEmpty(agent._action_history)
    self.assertIsNone(agent._observation)
    self.assertTrue(client.hide_automation_ui_called)
    self.assertTrue(client.reset_go_home)

  def test_step_data_shape_and_picklability(self):
    agent, _, _ = self._agent(_operation_response('TAP', '1'))
    action_data = agent.step('Tap Wi-Fi').data
    self.assertEqual(
        set(action_data),
        {'before_screenshot', 'after_screenshot', 'before_element_list',
         'action_prompt', 'action_output',
         'action_raw_response', 'summary_prompt', 'summary',
         'summary_raw_response', 'mobile_jev'},
    )
    self.assertNotIn('step_number', action_data)
    self.assertIsInstance(pickle.dumps(action_data), bytes)
    agent2, _, _ = self._agent(_operation_response('DONE'))
    done_data = agent2.step('Do nothing').data
    self.assertEqual(set(action_data), set(done_data))
    self.assertIsInstance(pickle.dumps(done_data), bytes)

  def test_action_is_recorded_in_history(self):
    agent, _, _ = self._agent(_operation_response('TAP', '1'))
    agent.step('Tap Wi-Fi')
    self.assertLen(agent._action_history, 1)
    entry = agent._action_history[0]
    self.assertEqual(entry.operation, 'TAP')
    self.assertEqual(entry.action['type'], 'tap_element')
    self.assertIsNotNone(entry.after)

  def test_repeated_action_is_stuck_without_dispatching(self):
    agent, client, _ = self._agent(_operation_response('TAP', '1'))
    first = agent.step('Tap Wi-Fi')
    self.assertFalse(first.done)
    second = agent.step('Tap Wi-Fi')
    self.assertTrue(second.done)
    self.assertEqual(second.data['mobile_jev']['status'], 'stuck')
    self.assertLen(client.taps, 1)

  def test_stale_decision_is_retried_once_for_real(self):
    initial = [_ui_element(text='Wi-Fi', clickable=True)]
    stale = [_ui_element(text='Wi-Fi changed', clickable=True)]
    client = FakeAndroidEnvClient(ui_elements=initial)
    # Reads: initial observe, stale pre-dispatch, fresh pre-dispatch, after.
    for elements in (initial, stale, stale, stale):
      client.queue_state(elements)
    agent, _, jev = _create_agent(
        client, _operation_response('TAP', '1')
    )
    result = agent.step('Tap Wi-Fi')
    self.assertFalse(result.done)
    self.assertLen(client.taps, 1)
    self.assertEqual(result.data['mobile_jev']['stale_retries'], 1)
    self.assertLen(jev.requests, 2)
    self.assertEqual(result.data['mobile_jev']['status'], 'action')

  def test_three_stale_decisions_end_the_episode(self):
    initial = [_ui_element(text='Wi-Fi', clickable=True)]
    client = FakeAndroidEnvClient(ui_elements=initial)
    client.queue_state(initial)
    for index in range(3):
      client.queue_state(
          [_ui_element(text=f'Changed {index}', clickable=True)]
      )
    agent, _, jev = _create_agent(
        client, _operation_response('TAP', '1')
    )
    result = agent.step('Tap Wi-Fi')
    self.assertTrue(result.done)
    self.assertEqual(
        result.data['mobile_jev']['status'], 'unstable_screen'
    )
    self.assertEqual(
        result.data['mobile_jev']['timings']['stale_retries'], 3
    )
    self.assertEmpty(client.executed_actions)
    self.assertEmpty(client.taps)
    self.assertLen(jev.requests, 3)

  def test_screen_size_change_between_reads_is_stale(self):
    client = FakeAndroidEnvClient(
        ui_elements=[_ui_element(text='Wi-Fi', clickable=True)],
        screen_size=(1000, 2000),
    )
    client.screen_size_queue = [
        (1000, 2000), (2000, 1000), (2000, 1000), (2000, 1000),
    ]
    agent, _, jev = _create_agent(client, _operation_response('TAP', '1'))
    result = agent.step('Tap Wi-Fi')
    self.assertFalse(result.done)
    self.assertEqual(result.data['mobile_jev']['stale_retries'], 1)
    self.assertLen(jev.requests, 2)
    self.assertLen(client.taps, 1)

  def test_installed_apps_are_refetched_after_a_failure(self):
    agent, client, _ = self._agent(_operation_response('DONE'))
    client.get_all_apps_failures = 1
    agent.step('Do nothing')
    self.assertIsNone(agent._installed_apps)
    agent.step('Do nothing')
    self.assertEqual(agent._installed_apps, ['Notes'])

  def test_non_stale_execution_error_propagates(self):
    agent, client, _ = self._agent(_operation_response('TAP', '1'))
    client.tap = mock.Mock(side_effect=RuntimeError('boom'))
    with self.assertRaisesRegex(RuntimeError, 'boom'):
      agent.step('Tap Wi-Fi')
    self.assertEmpty(agent._action_history)

  def test_action_is_kept_when_the_post_action_read_fails(self):
    agent, client, _ = self._agent(_operation_response('TAP', '1'))
    client.fail_get_state_after = 2
    with self.assertRaisesRegex(RuntimeError, 'state read failed'):
      agent.step('Tap Wi-Fi')
    self.assertLen(agent._action_history, 1)
    self.assertLen(client.taps, 1)

  def test_wait_is_local_and_bounded(self):
    agent, client, _ = self._agent(_operation_response('WAIT'))
    with mock.patch.object(mobile_jev.time, 'sleep') as sleep:
      result = agent.step('Wait')
      self.assertFalse(result.done)
      self.assertGreaterEqual(sleep.call_count, 1)
    self.assertEmpty(client.executed_actions)
    self.assertEmpty(client.taps)
    self.assertEqual(result.data['mobile_jev']['status'], 'action')

  def test_wait_budget_exhaustion_ends_the_episode(self):
    agent, client, _ = self._agent(
        _operation_response('WAIT'), wait_timeout_ms=0
    )
    with mock.patch.object(mobile_jev.time, 'sleep'):
      result = agent.step('Wait')
    self.assertTrue(result.done)
    self.assertEqual(
        result.data['mobile_jev']['status'], 'loading_timeout'
    )
    self.assertEmpty(client.executed_actions)
    self.assertEmpty(client.taps)

  def test_step_limit_does_not_end_the_episode(self):
    agent, client, _ = self._agent(_operation_response('TAP', '1'))
    agent.set_max_steps(1)
    agent.step('Tap Wi-Fi')
    result = agent.step('Tap Wi-Fi')
    self.assertFalse(result.done)
    self.assertEqual(result.data['mobile_jev']['status'], 'step_limit')
    self.assertLen(client.taps, 1)

  def test_decision_limit_stops_before_another_model_call(self):
    agent, _, jev = self._agent(_operation_response('TAP', '1'))
    agent.set_max_steps(1)
    agent._timings.model_calls = 2 * 1 + 4
    result = agent.step('Tap Wi-Fi')
    self.assertFalse(result.done)
    self.assertEqual(
        result.data['mobile_jev']['status'], 'decision_limit'
    )
    self.assertEmpty(jev.requests)

  def test_input_readback_verifies_and_records(self):
    def field(text):
      return [_ui_element(
          text=text,
          editable=True,
          focused=True,
          class_name='android.widget.EditText',
          resource_id='com.example:id/input',
          bounds=(0, 0, 500, 100),
      )]

    client = FakeAndroidEnvClient(ui_elements=field(''))
    # Reads: initial observe, pre-dispatch, after-action.
    client.queue_state(field(''))
    client.queue_state(field(''))
    client.queue_state(field('Tokyo'))
    agent, _, _ = _create_agent(client, _type_responder())
    with mock.patch.object(mobile_jev.time, 'sleep'):
      result = agent.step('Type Tokyo')
    self.assertFalse(result.done)
    self.assertEqual(result.data['mobile_jev']['status'], 'action')
    self.assertEqual(client.typed_texts, ['Tokyo'])

  def test_input_readback_failure_ends_the_episode(self):
    client = FakeAndroidEnvClient(ui_elements=[_ui_element(
        text='',
        editable=True,
        focused=True,
        class_name='android.widget.EditText',
        resource_id='com.example:id/input',
        bounds=(0, 0, 500, 100),
    )])
    agent, _, _ = _create_agent(
        client, _type_responder(), input_timeout_ms=0
    )
    with mock.patch.object(mobile_jev.time, 'sleep'):
      result = agent.step('Type Tokyo')
    self.assertTrue(result.done)
    self.assertEqual(
        result.data['mobile_jev']['status'], 'input_unverified'
    )
    self.assertEqual(client.typed_texts, ['Tokyo'])

  def test_recent_actions_are_capped_at_eight(self):
    elements = [
        _ui_element(
            text=f'Row {index}',
            clickable=True,
            bounds=(0, index * 60, 100, index * 60 + 50),
        )
        for index in range(9)
    ]
    client = FakeAndroidEnvClient(ui_elements=elements)
    calls = {'count': 0}

    def responder(request):
      calls['count'] += 1
      return _operation_response('TAP', str(calls['count']))(request)

    agent, _, jev = _create_agent(client, responder)
    for _ in range(9):
      agent.step('Tap each row')
    self.assertLen(agent._action_history, 9)
    self.assertLen(jev.requests[-1]['state']['recentActions'], 8)
    self.assertEqual(
        set(jev.requests[-1]['state']['recentActions'][0]),
        {'operation', 'label', 'screenChanged'},
    )


def _type_responder():
  def responder(request):
    questions = request['questions']
    criteria = questions['text_value']['criteria']
    key = next(k for k, v in criteria.items() if v == 'Tokyo')
    return _response({
        'operation': _answer(
            questions['operation']['criteria'], 'TYPE_TEXT'
        ),
        'text_value': _answer(criteria, key),
    })

  return responder


if __name__ == '__main__':
  absltest.main()
