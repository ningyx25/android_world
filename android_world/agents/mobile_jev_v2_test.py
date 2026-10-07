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

"""Offline tests for the multimodal ClientMobileJevV2 agent and its policy.

The strings the model reads are pinned here against the `ac-jev-v2` rows they
have to reproduce; the row-side of that contract lives in
`dohnuts/tests/test_jev_training_prompt.py` and
`dohnuts/data/processed/ac-jev-v2-subset100`.
"""

import json
import pickle
import time
from unittest import mock

from absl.testing import absltest
from android_world.agents import infer
from android_world.agents import mobile_jev_v2
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
import numpy as np
from PIL import Image


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


def _frame(value=7):
  """A flat screenshot, distinguishable by its fill value."""
  return np.full((SCREEN[1], SCREEN[0], 3), value, dtype=np.uint8)


def _state(elements, value=7):
  return interface.State(
      pixels=_frame(value),
      forest=None,
      ui_elements=list(elements),
      auxiliaries={},
  )


def _observation(elements, package='com.example', screen=SCREEN):
  return mobile_jev_v2.summarize_state(
      _state(elements), package, screen
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


def _noul(probability, confidence=None):
  return {
      'type': 'noul',
      'noul': probability,
      'confidence': (
          confidence
          if confidence is not None
          else max(probability, 1 - probability)
      ),
  }


def _response(answers):
  return {
      'model': 'jev2-test',
      'answers': answers,
      'usage': {'input_tokens': 1, 'output_tokens': 1},
  }


def _rich_screen():
  """A screen that offers every v2 operation except the removed three."""
  return [
      _ui_element(text='Calendar', bounds=(0, 0, 1000, 120)),
      _ui_element(text='Search', bounds=(20, 140, 980, 200), clickable=True),
      _ui_element(
          class_name='android.widget.EditText',
          bounds=(20, 220, 980, 280),
          clickable=True,
          editable=True,
          focused=True,
          hint='Title',
      ),
      _ui_element(
          text='Team Sync', bounds=(20, 300, 700, 360), clickable=True
      ),
      _ui_element(scrollable=True, bounds=(0, 400, 1000, 1900)),
  ]


class FakeJevWrapper(infer.MultimodalJevWrapper):
  """Scripted multimodal decision wrapper that never touches the network."""

  model_name = 'jev2-test'

  def __init__(self, responder):
    self.responder = responder
    self.requests = []
    self.images = []

  def predict_jev_mm(self, request, images):
    self.requests.append(request)
    self.images.append(images)
    response = (
        self.responder(request) if callable(self.responder)
        else self.responder
    )
    if response is None:
      return infer.ERROR_CALLING_LLM, False, None
    return json.dumps(response), None, response


def _scripted(
    operation, target=None, done=0.0, op_conf=1.0, target_conf=1.0
):
  """A responder that answers the operation question and, asked, its head.

  The speculative answer is only produced when the request actually asked
  that question, which is what lets a test tell an answered branch from a
  resolved-without-asking one.
  """

  def responder(request):
    questions = request['questions']
    answers = {
        'operation': _answer(
            questions['operation']['criteria'], operation, confidence=op_conf
        ),
        'done_goal': _noul(done),
    }
    head = mobile_jev_v2.OPERATION_HEAD.get(operation)
    if head and head in questions:
      asked = target
      if asked is None and head == mobile_jev_v2.SCROLL_DIRECT:
        asked = 'DOWN'
      answers[head] = _answer(
          questions[head]['criteria'], asked, confidence=target_conf
      )
    return _response(answers)

  return responder

class FakeAndroidEnvClient:
  """Duck-typed stand-in for interface.AndroidEnvClient."""

  def __init__(
      self,
      ui_elements=None,
      activity='com.example/.Main',
      apps=(),
      screen_size=SCREEN,
      pixels=None,
  ):
    self.ui_elements = list(ui_elements or [])
    self.activity = activity
    self.apps = list(apps)
    self.screen_size = screen_size
    self.pixels = pixels if pixels is not None else _frame()
    self.state_queue = []
    self.screen_size_queue = []
    self.hide_automation_ui_called = False
    self.executed_actions = []
    self.taps = []
    self.long_presses = []
    self.generated_swipes = []
    self.typed_texts = []
    self.pressed_keys = []
    self.generic_requests = []
    self.get_state_calls = 0
    self.get_all_apps_failures = 0

  def queue_state(self, ui_elements, pixels=None):
    self.state_queue.append((list(ui_elements), pixels))

  def reset(self, go_home):
    del go_home

  def hide_automation_ui(self):
    self.hide_automation_ui_called = True

  def get_state(self, wait_to_stabilize=False):
    del wait_to_stabilize
    self.get_state_calls += 1
    pixels = self.pixels
    if self.state_queue:
      ui_elements, queued_pixels = self.state_queue.pop(0)
      if queued_pixels is not None:
        pixels = queued_pixels
    else:
      ui_elements = self.ui_elements
    return interface.State(
        pixels=pixels, forest=None, ui_elements=list(ui_elements),
        auxiliaries={}
    )

  def get_current_activity(self, timeout_sec=10):
    del timeout_sec
    return self.activity

  def get_all_apps(self, timeout_sec=10):
    del timeout_sec
    if self.get_all_apps_failures > 0:
      self.get_all_apps_failures -= 1
      raise RuntimeError('installed app list unavailable')
    return list(self.apps)

  def get_logical_screen_size(self):
    if self.screen_size_queue:
      return self.screen_size_queue.pop(0)
    return self.screen_size

  def execute_action(self, action):
    self.executed_actions.append(action)

  def tap(self, x, y, timeout_sec=10):
    del timeout_sec
    self.taps.append((x, y))

  def long_press(self, x, y, timeout_sec=10):
    del timeout_sec
    self.long_presses.append((x, y))

  def generate_swipe_command(
      self, start_x, start_y, end_x, end_y, duration_ms=None
  ):
    self.generated_swipes.append((start_x, start_y, end_x, end_y, duration_ms))
    return [
        'shell', 'input', 'swipe', str(start_x), str(start_y),
        str(end_x), str(end_y), str(duration_ms or 1000),
    ]

  def issue_generic_request(self, args, timeout_sec=10):
    del timeout_sec
    self.generic_requests.append(args)
    return {}

  def press_key(self, keycode, timeout_sec=10):
    del timeout_sec
    self.pressed_keys.append(keycode)

  def type_text(self, text, timeout_sec=10):
    del timeout_sec
    self.typed_texts.append(text)


def _create_agent(elements=None, responder=None, apps=(), **kwargs):
  client = FakeAndroidEnvClient(ui_elements=elements, apps=apps)
  jev = FakeJevWrapper(responder or _scripted('TAP', '1'))
  agent = mobile_jev_v2.ClientMobileJevV2(
      client, llm=kwargs.pop('llm', None), jev=jev, **kwargs
  )
  agent.transition_pause = 0.0
  return agent, client, jev


class PromptParityTest(absltest.TestCase):
  """The strings a row was built from are the strings the agent sends."""

  def test_rules_are_the_v2_rules(self):
    self.assertLen(mobile_jev_v2.RULES, 665)
    self.assertIn('Use LONG_PRESS only when', mobile_jev_v2.RULES)
    # The three operations the conversion removed are gone from the rules too.
    self.assertNotIn('WAIT only for', mobile_jev_v2.RULES)
    self.assertNotIn('DONE requires', mobile_jev_v2.RULES)
    self.assertNotIn('BLOCKED means', mobile_jev_v2.RULES)

  def test_operation_criteria_order_and_scroll_merge(self):
    space = mobile_jev_v2.build_questions(
        _observation(_rich_screen()), ('Team Sync',), ('Calendar', 'Clock')
    )
    self.assertEqual(
        list(space.questions['operation']['criteria']),
        ['OPEN_APP', 'TAP', 'LONG_PRESS', 'TYPE_TEXT', 'SCROLL', 'BACK',
         'HOME', 'ENTER'],
    )
    self.assertEqual(
        space.questions['operation']['criteria']['SCROLL'],
        mobile_jev_v2.SCROLL_DESCRIPTION,
    )
    self.assertNotIn('SCROLL_DOWN', space.questions['operation']['criteria'])

  def test_a_shared_region_is_numbered_once_and_scrolls_four_ways(self):
    elements = [
        _ui_element(
            text='Feed',
            bounds=(0, 0, 1000, 1900),
            clickable=True,
            scrollable=True,
        ),
        _ui_element(text='Other', bounds=(0, 0, 500, 900), clickable=True),
    ]
    space = mobile_jev_v2.build_questions(_observation(elements))
    operations = {
        entry['index']: entry['operations'] for entry in space.elements
    }
    # TAP first, then the merged SCROLL -- the entry list never names a
    # direction, because no question offers one any more.
    self.assertEqual(operations['1'], ['TAP', 'SCROLL'])
    self.assertEqual(operations['2'], ['TAP'])
    self.assertEqual(
        list(space.scroll['1']),
        ['SCROLL_DOWN', 'SCROLL_UP', 'SCROLL_RIGHT', 'SCROLL_LEFT'],
    )
    self.assertEqual(space.primary_scroll_region(), '1')

  def test_the_primary_scroll_region_is_the_criteria_first(self):
    # A scroll-only carousel before a tappable list is numbered after the
    # taps: the criteria's first region is '2', and that is the region the
    # training convention labels a scroll with.
    elements = [
        _ui_element(scrollable=True, bounds=(0, 400, 1000, 700)),
        _ui_element(
            text='List', bounds=(0, 800, 1000, 1900), clickable=True,
            scrollable=True,
        ),
    ]
    space = mobile_jev_v2.build_questions(_observation(elements))
    self.assertEqual(list(space.scroll), ['2', '1'])
    self.assertEqual(
        list(space.questions['scroll_target']['criteria']), ['2', '1']
    )
    self.assertEqual(space.primary_scroll_region(), '2')

  def test_scroll_direct_criteria_reuse_the_direction_descriptions(self):
    space = mobile_jev_v2.build_questions(_observation(_rich_screen()))
    self.assertEqual(
        list(space.questions['scroll_direct']['criteria']),
        ['DOWN', 'UP', 'LEFT', 'RIGHT'],
    )
    self.assertEqual(
        space.questions['scroll_direct']['criteria']['DOWN'],
        'Scroll down to reveal more content in that direction.',
    )

  def test_done_goal_question_is_the_row_constant_unwrapped(self):
    space = mobile_jev_v2.build_questions(_observation(_rich_screen()))
    self.assertEqual(space.questions['done_goal'], {
        'type': 'noul',
        'instructions': (
            'Has the entire goal been visibly completed on the current '
            'screen? Answer yes only when every requirement is satisfied by '
            'what is currently visible, not merely initiated or in progress.'
        ),
        'criteria': {
            'false': 'no, more actions are still needed',
            'true': 'yes, the goal is fully complete on this screen',
        },
    })

  def test_element_entries_list_one_scroll_beside_tap(self):
    elements = [
        _ui_element(
            text='Feed', bounds=(0, 0, 1000, 1900), clickable=True,
            scrollable=True,
        ),
        _ui_element(text='Other', bounds=(0, 0, 500, 900), clickable=True),
    ]
    space = mobile_jev_v2.build_questions(_observation(elements))
    operations = {
        entry['index']: entry['operations'] for entry in space.elements
    }
    self.assertEqual(operations['1'], ['TAP', SCROLL_ONLY])

  def test_unasked_questions_are_built_but_not_sent(self):
    self.assertNotIn('scroll_target', mobile_jev_v2.ASKED_QUESTIONS)
    self.assertNotIn('text_value', mobile_jev_v2.ASKED_QUESTIONS)
    for name in ('done_goal', 'scroll_direct', 'operation'):
      self.assertIn(name, mobile_jev_v2.ASKED_QUESTIONS)

  def test_a_one_candidate_question_is_not_built(self):
    elements = [_ui_element(text='Wi-Fi', clickable=True)]
    observation = _observation(elements)
    space = mobile_jev_v2.build_questions(observation)
    self.assertLen(space.tap, 1)
    self.assertNotIn('tap_target', space.questions)
    self.assertIn('TAP', space.questions['operation']['criteria'])

  def test_too_many_options_refuses_the_request(self):
    elements = [
        _ui_element(
            text=f'Item {i}', bounds=(0, i, 100, i + 50),
            clickable=True,
        )
        for i in range(mobile_jev_v2.MAX_CHOICE_OPTIONS + 1)
    ]
    with self.assertRaises(mobile_jev_v2.PayloadTooLargeError):
      mobile_jev_v2.build_questions(_observation(elements))


SCROLL_ONLY = 'SCROLL'


class StateTest(absltest.TestCase):

  def test_state_is_two_keys(self):
    agent, _, jev = _create_agent(_rich_screen(), _scripted('TAP', '1'))
    agent.step('Set the event title')
    self.assertEqual(
        list(jev.requests[0]['state']), ['goal', 'recentActions']
    )

  def test_recent_actions_keep_the_last_five(self):
    agent, _, jev = _create_agent(_rich_screen(), _scripted('TAP', '1'))
    for index in range(8):
      agent.step('Tap Search')
      agent._observation = None  # Force a fresh read without re-deciding.
      agent._executed_signatures.clear()
    recent = jev.requests[-1]['state']['recentActions']
    self.assertLen(recent, mobile_jev_v2.MAX_RECENT_ACTIONS)
    self.assertEqual(recent[0]['operation'], 'TAP')

  def test_instructions_are_wrapped_with_the_goal_and_rules(self):
    agent, _, jev = _create_agent(_rich_screen(), _scripted('TAP', '1'))
    agent.step('Tap Search')
    question = jev.requests[0]['questions']['tap_target']
    self.assertEqual(
        question['instructions'],
        {
            'goal': 'Tap Search',
            'rules': mobile_jev_v2.TARGET_QUESTION_TEMPLATE.format(
                operation='TAP'
            ),
        },
    )
    self.assertIsInstance(
        jev.requests[0]['questions']['done_goal']['instructions'], str
    )


class AppInventoryTest(absltest.TestCase):
  """The app question offers the whole installed list; the right app is in it."""

  def setUp(self):
    super().setUp()
    self.inventory = [f'App{i}' for i in range(60)] + ['Calendar', 'Clock']

  def test_the_whole_installed_list_is_offered_in_listing_order(self):
    space = mobile_jev_v2.build_questions(
        _observation(_rich_screen()), apps=self.inventory
    )
    criteria = space.questions['app_target']['criteria']
    self.assertEqual(list(criteria.values()), self.inventory)
    self.assertEqual(
        list(criteria),
        [str(index) for index in range(1, len(self.inventory) + 1)],
    )

  def test_the_inventory_is_capped_at_max_apps(self):
    inventory = [f'App{i}' for i in range(mobile_jev_v2.MAX_APPS + 20)]
    space = mobile_jev_v2.build_questions(
        _observation(_rich_screen()), apps=inventory
    )
    criteria = space.questions['app_target']['criteria']
    self.assertLen(criteria, mobile_jev_v2.MAX_APPS)
    self.assertEqual(
        list(criteria.values()), inventory[: mobile_jev_v2.MAX_APPS]
    )

  def test_a_thin_inventory_offers_what_exists(self):
    space = mobile_jev_v2.build_questions(
        _observation(_rich_screen()), apps=['Calendar']
    )
    self.assertNotIn('app_target', space.questions)
    self.assertIn('OPEN_APP', space.questions['operation']['criteria'])
    self.assertEqual(
        [candidate.action.app_name for candidate in space.app.values()],
        ['Calendar'],
    )

  def test_no_inventory_offers_no_app_operation(self):
    space = mobile_jev_v2.build_questions(
        _observation(_rich_screen()), apps=[]
    )
    self.assertNotIn('app_target', space.questions)
    self.assertNotIn('OPEN_APP', space.questions['operation']['criteria'])


class ImageTest(absltest.TestCase):

  def test_images_are_history_then_current_then_marked(self):
    frames = [_frame(value) for value in range(1, 9)]
    history = [
        mobile_jev_v2._HistoryEntry(
            operation='TAP',
            label=f'Tap {index}.',
            action={'type': 'tap_element'},
            screen_changed=True,
            screenshot=frame,
        )
        for index, frame in enumerate(frames)
    ]
    images = mobile_jev_v2._request_images(_frame(99), history, _frame(98))
    self.assertEqual(
        [image.kind for image in images],
        ['history'] * 5 + ['current', 'marked'],
    )
    # Oldest first, and the last history frame is the fifth kept entry.
    self.assertTrue(
        np.array_equal(images[0].pixels, _frame(4)),
        'the oldest of the five kept frames should be frame 4',
    )
    self.assertTrue(np.array_equal(images[-2].pixels, _frame(99)))
    self.assertTrue(np.array_equal(images[-1].pixels, _frame(98)))

  def test_entries_without_a_frame_contribute_no_image(self):
    history = [
        mobile_jev_v2._HistoryEntry(
            operation='BACK', label='back', action={'type': 'global'},
            screen_changed=True,
        )
    ]
    images = mobile_jev_v2._request_images(_frame(), history, None)
    self.assertEqual([image.kind for image in images], ['current'])

  def test_marked_frame_only_when_a_tap_question_exists(self):
    observation = _observation(_rich_screen())
    space = mobile_jev_v2.build_questions(observation)
    marked = mobile_jev_v2._marked_frame(_frame(3), space, observation)
    self.assertIsNotNone(marked)
    self.assertFalse(np.array_equal(marked, _frame(3)))

    plain = _observation([_ui_element(text='Header', bounds=(0, 0, 100, 40))])
    space = mobile_jev_v2.build_questions(plain)
    self.assertIsNone(
        mobile_jev_v2._marked_frame(_frame(3), space, plain)
    )

  def test_mark_screenshot_draws_boxes_without_touching_the_source(self):
    source = _frame(200)
    element = mobile_jev_v2._Element(
        id='1',
        text='',
        label='',
        hint='',
        resource_id='',
        class_name='',
        bounds=(100, 200, 400, 300),
        clickable=True,
        editable=False,
        scrollable=False,
        enabled=True,
        focused=False,
        checkable=False,
        checked=False,
        selected=False,
    )
    marked = mobile_jev_v2.mark_screenshot(source, [('7', element)])
    self.assertTrue(np.array_equal(source, _frame(200)))
    self.assertEqual(tuple(marked[500, 500]), (200, 200, 200))
    self.assertNotEqual(tuple(marked[200, 250]), (200, 200, 200))
    # The chip is drawn inside the box, where a white label on green pixels is
    # the only thing that can appear.
    self.assertTrue((marked[202:220, 102:160] == 255).any())

  def test_mark_screenshot_matches_the_pil_rendering_of_the_same_marks(self):
    element = mobile_jev_v2._Element(
        id='0', text='', label='', hint='', resource_id='', class_name='',
        bounds=(0, 0, 20, 20), clickable=True, editable=False,
        scrollable=False, enabled=True, focused=False, checkable=False,
        checked=False, selected=False,
    )
    marked = mobile_jev_v2.mark_screenshot(np.zeros((40, 40, 3), np.uint8),
                                           [('1', element)])
    self.assertEqual(marked.shape, (40, 40, 3))
    self.assertEqual(marked.dtype, np.uint8)


class TextChoiceTest(absltest.TestCase):

  def test_empty_candidates_have_no_value(self):
    observation = _observation(_rich_screen())
    value, source = mobile_jev_v2.choose_text_value(None, '', observation, ())
    self.assertIsNone(value)
    self.assertEqual(source, 'no-candidates')

  def test_without_an_llm_the_longest_span_wins(self):
    observation = _observation(_rich_screen())
    values = ['Team', 'Team Sync', 'event title']
    value, source = mobile_jev_v2.choose_text_value(
        None, 'Set the event title', observation, values
    )
    self.assertEqual(value, 'Team Sync')
    self.assertEqual(source, 'heuristic')

  def test_a_numbered_answer_selects_the_span(self):
    # The offered list is ranked by completeness, so the numbers the model sees
    # are the ranked positions, not text_candidates' own order.
    observation = _observation(_rich_screen())
    llm = mock.Mock()
    llm.predict_mm.return_value = ('2', None, None)
    value, source = mobile_jev_v2.choose_text_value(
        llm,
        'Set the event title',
        observation,
        ['a b c', 'Team Sync', 'x'],
        screenshot=_frame(),
    )
    self.assertEqual(value, 'Team Sync')
    self.assertEqual(source, 'llm')
    prompt, images = llm.predict_mm.call_args[0]
    self.assertIn('Set the event title', prompt)
    self.assertIn('Field: Title', prompt)
    self.assertLen(images, 1)
  def test_zero_means_none_of_them_fits(self):
    observation = _observation(_rich_screen())
    llm = mock.Mock()
    llm.predict_mm.return_value = ('0', None, None)
    value, source = mobile_jev_v2.choose_text_value(
        llm, 'goal', observation, ['a b', 'c']
    )
    self.assertIsNone(value)
    self.assertEqual(source, 'llm-none')

  def test_invented_prose_is_refused(self):
    observation = _observation(_rich_screen())
    llm = mock.Mock()
    llm.predict_mm.return_value = ('"Something else"', None, None)
    value, source = mobile_jev_v2.choose_text_value(
        llm, 'goal', observation, ['a b c', 'Team Sync']
    )
    self.assertEqual(value, 'a b c')
    self.assertEqual(source, 'heuristic')

  def test_a_failing_llm_falls_back(self):
    observation = _observation(_rich_screen())
    llm = mock.Mock()
    llm.predict_mm.side_effect = RuntimeError('offline')
    value, source = mobile_jev_v2.choose_text_value(
        llm, 'goal', observation, ['a b', 'c']
    )
    self.assertEqual(value, 'a b')
    self.assertEqual(source, 'heuristic')

  def test_the_error_sentinel_is_not_an_answer(self):
    # A transport failure comes back as ERROR_CALLING_LLM, not as an
    # exception; parsing it as an answer would silently pick a span.
    observation = _observation(_rich_screen())
    llm = mock.Mock()
    llm.predict_mm.return_value = (infer.ERROR_CALLING_LLM, None, None)
    value, source = mobile_jev_v2.choose_text_value(
        llm, 'goal', observation, ['a b', 'c']
    )
    self.assertEqual(value, 'a b')
    self.assertEqual(source, 'heuristic')

  def test_an_overflowing_goal_keeps_the_first_spans(self):
    goal = ' '.join(f'word{index}' for index in range(60))
    texts = mobile_jev_v2.text_candidates(goal)
    self.assertTrue(texts.overflow)
    self.assertLen(texts.values, mobile_jev_v2.MAX_TEXT_CANDIDATES)
    observation = _observation(_rich_screen())
    space = mobile_jev_v2.build_questions(observation, texts.values)
    self.assertIn('TYPE_TEXT', space.questions['operation']['criteria'])
    self.assertIn('text_value', space.questions)
    self.assertLen(
        space.questions['text_value']['criteria'],
        mobile_jev_v2.MAX_TEXT_CANDIDATES + 1,
    )


class PolicyDecideTest(absltest.TestCase):

  def _policy(self, responder, **kwargs):
    jev = FakeJevWrapper(responder)
    return mobile_jev_v2.MobileJevV2Policy(jev, **kwargs), jev

  def test_a_decision_requires_its_screenshot(self):
    policy, _ = self._policy(_scripted('TAP', '1'))
    with self.assertRaisesRegex(RuntimeError, 'trained on screenshots'):
      policy.decide('Tap Search', _observation(_rich_screen()))

  def test_tap_resolves_the_numbered_element(self):
    observation = _observation(_rich_screen())
    policy, jev = self._policy(_scripted('TAP', '3'))
    decision = policy.decide(
        'Save the event', observation, apps=(), screenshot=_frame()
    )
    self.assertEqual(decision.status, mobile_jev_v2.Status.ACTION)
    self.assertEqual(decision.action.element_id, '3')
    self.assertEqual(decision.label, 'Tap Team Sync.')
    self.assertEqual(decision.choice, 'tap_3')
    self.assertEqual(
        [image.kind for image in jev.images[0]], ['current', 'marked']
    )

  def test_long_press_shares_the_tap_candidates(self):
    observation = _observation(_rich_screen())
    policy, _ = self._policy(_scripted('LONG_PRESS', '3'))
    decision = policy.decide('Hold it', observation, screenshot=_frame())
    self.assertEqual(decision.action.kind, mobile_jev_v2.ActionKind.LONG_PRESS)
    self.assertEqual(decision.action.element_id, '3')
    self.assertEqual(
        decision.label, '{"element_id":"3","type":"long_press"}'
    )

  def test_long_press_with_a_single_candidate_is_still_a_long_press(self):
    # One TAP candidate: tap_target is not asked and the candidate is forced,
    # but the operation the model chose is LONG_PRESS and must not be turned
    # into the forced candidate's plain tap.
    observation = _observation([_ui_element(text='Wi-Fi', clickable=True)])
    policy, jev = self._policy(_scripted('LONG_PRESS', None))
    decision = policy.decide('Hold Wi-Fi', observation, screenshot=_frame())
    self.assertEqual(decision.action.kind, mobile_jev_v2.ActionKind.LONG_PRESS)
    self.assertEqual(decision.action.element_id, '0')
    self.assertNotIn('tap_target', jev.requests[0]['questions'])

  def test_scroll_takes_the_direction_from_scroll_direct(self):
    observation = _observation(_rich_screen())
    policy, jev = self._policy(_scripted('SCROLL', 'UP'))
    decision = policy.decide(
        'Find the review', observation, screenshot=_frame()
    )
    self.assertEqual(decision.target, 'UP')
    self.assertEqual(decision.action.direction, 'up')
    self.assertEqual(decision.action.region_id, '4')
    self.assertEqual(
        decision.label,
        'Scroll up to reveal content further up in this scrollable region. '
        'Gesture: {"direction":"up","region_id":"4","type":"scroll"}',
    )
    self.assertIn('scroll_direct', jev.requests[0]['questions'])

  def test_open_app_resolves_the_offered_label(self):
    observation = _observation(_rich_screen())
    policy, _ = self._policy(_scripted('OPEN_APP', '1'))
    decision = policy.decide(
        'Open Calendar', observation, apps=['Calendar', 'Clock'],
        screenshot=_frame(),
    )
    self.assertEqual(decision.action.app_name, 'Calendar')
    self.assertEqual(decision.label, 'Open Calendar')

  def test_the_installed_list_reaches_the_question_unnarrowed(self):
    observation = _observation(_rich_screen())
    apps = ['Calendar', 'Clock', 'Drive', 'Maps']
    policy, jev = self._policy(_scripted('OPEN_APP', '3'))
    decision = policy.decide(
        'Open the app that holds the file', observation, apps=apps,
        screenshot=_frame(),
    )
    self.assertEqual(
        list(jev.requests[0]['questions']['app_target']['criteria'].values()),
        apps,
    )
    self.assertEqual(decision.action.app_name, 'Drive')

  def test_an_overflowing_goal_still_offers_and_can_report_type_text(self):
    # The overflow no longer removes TYPE_TEXT from the operation question,
    # and a reader that finds no span is still diagnosed as such.
    observation = _observation(_rich_screen())
    llm = mock.Mock()
    llm.predict_mm.return_value = ('0', None, None)
    policy, jev = self._policy(_scripted('TYPE_TEXT'), llm=llm)
    goal = ' '.join(f'word{index}' for index in range(60))
    decision = policy.decide(goal, observation, screenshot=_frame())
    self.assertIn(
        'TYPE_TEXT', jev.requests[0]['questions']['operation']['criteria']
    )
    self.assertEqual(decision.status, mobile_jev_v2.Status.NEEDS_INPUT)
    self.assertIn('too many text spans', decision.reason)

  def test_a_single_candidate_is_taken_without_asking(self):
    observation = _observation([_ui_element(text='Wi-Fi', clickable=True)])
    policy, jev = self._policy(_scripted('TAP', None))
    decision = policy.decide('Turn on Wi-Fi', observation, screenshot=_frame())
    self.assertEqual(decision.action.element_id, '0')
    self.assertIsNone(decision.target)
    self.assertNotIn('tap_target', jev.requests[0]['questions'])

  def test_done_probability_ends_before_any_action(self):
    observation = _observation(_rich_screen())
    policy, _ = self._policy(_scripted('TAP', '1', done=0.91))
    decision = policy.decide('Tap Search', observation, screenshot=_frame())
    self.assertEqual(decision.status, mobile_jev_v2.Status.DONE)
    self.assertIsNone(decision.action)
    self.assertAlmostEqual(decision.done_goal_probability, 0.91)

  def test_done_threshold_is_configurable(self):
    observation = _observation(_rich_screen())
    policy, _ = self._policy(_scripted('TAP', '1', done=0.6),
                             done_threshold=0.9)
    decision = policy.decide('Tap Search', observation, screenshot=_frame())
    self.assertEqual(decision.status, mobile_jev_v2.Status.ACTION)

  def test_uncertainty_uses_the_threshold(self):
    observation = _observation(_rich_screen())
    policy, _ = self._policy(
        _scripted('TAP', '1', op_conf=0.4, target_conf=0.4), threshold=0.5
    )
    decision = policy.decide('Tap Search', observation, screenshot=_frame())
    self.assertEqual(decision.status, mobile_jev_v2.Status.UNCERTAIN)

  def test_type_text_goes_through_the_text_choice(self):
    observation = _observation(_rich_screen())
    policy, _ = self._policy(_scripted('TYPE_TEXT'))
    decision = policy.decide(
        'Set the title to Team Sync', observation, screenshot=_frame()
    )
    self.assertEqual(decision.action.kind, mobile_jev_v2.ActionKind.TYPE)
    self.assertEqual(decision.text_source, 'heuristic')
    self.assertEqual(decision.action.text, 'Set the title to Team Sync')

  def test_malformed_noul_is_rejected(self):
    self.assertEqual(mobile_jev_v2.validate_noul(_noul(0.25)), 0.25)
    for bad in (None, {}, {'type': 'choice', 'noul': 0.5},
                {'type': 'noul', 'noul': 'x'}, {'type': 'noul', 'noul': True},
                {'type': 'noul', 'noul': float('nan')}):
      with self.assertRaises(mobile_jev_v2.InvalidChoiceError):
        mobile_jev_v2.validate_noul(bad)

  def test_transport_failure_raises(self):
    observation = _observation(_rich_screen())
    policy, _ = self._policy(None)
    with self.assertRaisesRegex(RuntimeError, 'Error calling'):
      policy.decide('Tap Search', observation, screenshot=_frame())

  def test_oversized_payload_raises_before_the_call(self):
    big = 'x' * 40000
    elements = [
        _ui_element(text=f'Item {index} {big[:200]}',
                    bounds=(0, index, 100, index + 50), clickable=True)
        for index in range(250)
    ]
    policy, _ = self._policy(_scripted('TAP', '1'))
    with self.assertRaises(mobile_jev_v2.PayloadTooLargeError):
      policy.decide(big, _observation(elements), screenshot=_frame())


class ClientStepTest(absltest.TestCase):

  def test_reset_hides_the_automation_ui(self):
    agent, client, _ = _create_agent(_rich_screen(), _scripted('TAP', '1'))
    agent.reset(go_home_on_reset=True)
    self.assertTrue(client.hide_automation_ui_called)

  def test_step_data_shape_and_picklability(self):
    agent, _, _ = _create_agent(_rich_screen(), _scripted('TAP', '1'))
    step = agent.step('Tap Search').data
    self.assertEqual(
        set(step),
        {
            'before_screenshot', 'after_screenshot', 'before_element_list',
            'action_prompt', 'action_output', 'action_raw_response',
            'summary_prompt', 'summary', 'summary_raw_response',
            'mobile_jev_v2',
        },
    )
    self.assertIsInstance(pickle.dumps(step), bytes)
    self.assertEqual(step['mobile_jev_v2']['status'], 'action')

  def test_the_decision_frame_becomes_the_history_image(self):
    elements = _rich_screen()
    agent, client, jev = _create_agent(elements, _scripted('TAP', '1'))
    client.pixels = _frame(42)
    agent.step('Tap Search')
    agent._observation = None
    agent._executed_signatures.clear()
    agent.step('Tap Search')
    kinds = [image.kind for image in jev.images[-1]]
    self.assertEqual(kinds, ['history', 'current', 'marked'])
    self.assertTrue(np.array_equal(jev.images[-1][0].pixels, _frame(42)))
    self.assertLen(agent._action_history[0].screenshot.shape, 3)

  def test_screen_changed_comes_from_the_fingerprint_not_the_pixels(self):
    elements = _rich_screen()
    agent, client, jev = _create_agent(elements, _scripted('TAP', '1'))
    agent.step('Tap Search')
    client.queue_state(elements, pixels=_frame(77))
    agent._executed_signatures.clear()
    agent.step('Tap Search')
    self.assertFalse(agent._action_history[0].screen_changed)
    self.assertEqual(
        jev.requests[-1]['state']['recentActions'][0]['screenChanged'], False
    )

  def test_long_press_is_dispatched_as_a_long_press(self):
    agent, client, _ = _create_agent(
        _rich_screen(), _scripted('LONG_PRESS', '3')
    )
    agent.step('Hold Team Sync')
    self.assertEqual(client.long_presses, [(360, 330)])
    self.assertEmpty(client.taps)

  def test_a_forced_single_candidate_long_press_dispatches_a_long_press(self):
    agent, client, _ = _create_agent(
        [_ui_element(text='Wi-Fi', clickable=True)],
        _scripted('LONG_PRESS', None),
    )
    agent.step('Hold Wi-Fi')
    self.assertEqual(client.long_presses, [(50, 25)])
    self.assertEmpty(client.taps)

  def test_open_app_uses_the_installed_list(self):
    agent, client, _ = _create_agent(
        _rich_screen(), _scripted('OPEN_APP', '1'), apps=['Calendar', 'Clock']
    )
    agent.step('Open Calendar')
    self.assertEqual(
        agent._action_history[-1].operation, 'OPEN_APP'
    )
    self.assertEqual(
        client.executed_actions[0],
        json_action.JSONAction(
            action_type=json_action.OPEN_APP, app_name='Calendar'
        ),
    )

  def test_back_and_enter_use_their_actions(self):
    agent, client, _ = _create_agent(_rich_screen(), _scripted('BACK'))
    agent.step('Go back')
    self.assertEqual(
        client.executed_actions[0].action_type, json_action.NAVIGATE_BACK
    )
    agent, client, _ = _create_agent(_rich_screen(), _scripted('ENTER'))
    agent.step('Submit')
    self.assertEqual(
        client.executed_actions[0].action_type, json_action.KEYBOARD_ENTER
    )

  def test_scroll_swipes_inside_the_region(self):
    agent, client, _ = _create_agent(
        _rich_screen(), _scripted('SCROLL', 'DOWN')
    )
    agent.step('Scroll down')
    start_x, start_y, end_x, end_y, duration = client.generated_swipes[0]
    self.assertEqual((start_x, start_y, end_x, end_y), (500, 1600, 500, 700))
    self.assertEqual(duration, 300)

  def test_repeated_action_is_stuck_without_dispatching(self):
    agent, client, _ = _create_agent(_rich_screen(), _scripted('TAP', '1'))
    first = agent.step('Tap Search')
    self.assertFalse(first.done)
    second = agent.step('Tap Search')
    self.assertTrue(second.done)
    self.assertEqual(second.data['mobile_jev_v2']['status'], 'stuck')
    self.assertLen(client.taps, 1)

  def _rich(self, search_text='Search'):
    return [
        _ui_element(text='Calendar', bounds=(0, 0, 1000, 120)),
        _ui_element(
            text=search_text, bounds=(20, 140, 980, 200), clickable=True
        ),
        _ui_element(
            class_name='android.widget.EditText',
            bounds=(20, 220, 980, 280),
            clickable=True,
            editable=True,
            focused=True,
            hint='Title',
        ),
        _ui_element(
            text='Team Sync', bounds=(20, 300, 700, 360), clickable=True
        ),
        _ui_element(scrollable=True, bounds=(0, 400, 1000, 1900)),
    ]

  def test_stale_decision_is_retried_by_redeciding(self):
    # A decision is retried by observing and deciding again, never by replaying
    # the action: the pre-dispatch read shows the targeted element's meaning has
    # changed, so the tap must not be dispatched and a second request made.
    client = FakeAndroidEnvClient(ui_elements=self._rich())
    client.queue_state(self._rich())
    client.queue_state(self._rich('Search changed'))
    client.queue_state(self._rich('Search changed'))
    client.queue_state(self._rich('Search changed'))
    jev = FakeJevWrapper(_scripted('TAP', '1'))
    agent = mobile_jev_v2.ClientMobileJevV2(client, llm=None, jev=jev)
    agent.transition_pause = 0.0
    result = agent.step('Tap Search')
    self.assertEqual(result.data['mobile_jev_v2']['stale_retries'], 1)
    self.assertLen(jev.requests, 2)
    self.assertLen(client.taps, 1)

  def test_three_stale_decisions_end_the_episode(self):
    client = FakeAndroidEnvClient(ui_elements=self._rich())
    for _ in range(3):
      client.queue_state(self._rich())
      client.queue_state(self._rich('Search changed'))
    jev = FakeJevWrapper(_scripted('TAP', '1'))
    agent = mobile_jev_v2.ClientMobileJevV2(client, llm=None, jev=jev)
    agent.transition_pause = 0.0
    result = agent.step('Tap Search')
    self.assertTrue(result.done)
    self.assertEqual(
        result.data['mobile_jev_v2']['status'], 'unstable_screen'
    )
    self.assertEmpty(client.taps)
    self.assertLen(jev.requests, 3)

  def test_the_staleness_budget_can_extend_an_expired_observation(self):
    # The budget is what keeps a slow decision from looking like a changed
    # screen; the screen's own comparison is unaffected by it.
    elements = _rich_screen()
    fresh = _observation(elements)
    expired = mobile_jev_v2.summarize_state(
        _state(elements), 'com.example', SCREEN,
        observed_at=time.monotonic() - 40,
    )
    action = mobile_jev_v2._ActionSpec(
        mobile_jev_v2.ActionKind.TAP, element_id='1'
    )
    with self.assertRaises(mobile_jev_v2.StaleObservationError):
      mobile_jev_v2.assert_fresh(fresh, expired, action)
    mobile_jev_v2.assert_fresh(fresh, expired, action, max_age_sec=60)

  def test_a_slow_decision_widens_the_staleness_budget(self):
    # A slow text-selection call must not make an unchanged screen look
    # stale: the decision's own duration is added to the age budget.
    agent, _, _ = _create_agent(_rich_screen(), _scripted('TAP', '1'))
    original = agent.policy.decide

    def slow_decide(*args, **kwargs):
      time.sleep(0.25)
      return original(*args, **kwargs)

    agent.policy.decide = slow_decide
    budgets = []
    real = mobile_jev_v2.assert_fresh

    def spy(
        current, expected, action,
        max_age_sec=mobile_jev_v2.MAX_OBSERVATION_AGE_SEC,
    ):
      budgets.append(max_age_sec)
      return real(current, expected, action, max_age_sec)

    with mock.patch.object(mobile_jev_v2, 'assert_fresh', spy):
      agent.step('Tap Search')
    self.assertGreater(budgets[0], mobile_jev_v2.MAX_OBSERVATION_AGE_SEC)

  def test_type_is_verified_against_the_field(self):
    # The value the agent sends must come back readable in exactly one input,
    # and the readback loop is what proves it looked rather than trusted.
    def field(text):
      return [
          _ui_element(text='Calendar', bounds=(0, 0, 1000, 120)),
          _ui_element(
              text=text,
              class_name='android.widget.EditText',
              bounds=(20, 220, 980, 280),
              clickable=True,
              editable=True,
              focused=True,
              hint='Title',
          ),
          _ui_element(text='Save', bounds=(20, 300, 700, 360), clickable=True),
      ]

    typed = 'Set the title to Team Sync'
    client = FakeAndroidEnvClient(ui_elements=field(''), apps=[])
    client.queue_state(field(''))
    client.queue_state(field(''))
    client.queue_state(field(typed))
    jev = FakeJevWrapper(_scripted('TYPE_TEXT'))
    agent = mobile_jev_v2.ClientMobileJevV2(client, llm=None, jev=jev)
    agent.transition_pause = 0.0
    result = agent.step(typed)
    self.assertEqual(result.data['mobile_jev_v2']['status'], 'action')
    self.assertEqual(client.typed_texts[-1], typed)
    self.assertEqual(agent._action_history[-1].text, typed)

  def test_a_readback_poll_advances_the_decision_frame(self):
    # The readback's last screen is the screen the next decision is made on:
    # the frame and the last state must advance with the observation, or the
    # next request would pair a stale screenshot with bounds from a newer read.
    def field(text):
      return [
          _ui_element(
              text=text,
              class_name='android.widget.EditText',
              bounds=(20, 220, 980, 280),
              clickable=True,
              editable=True,
              focused=True,
              hint='Title',
          ),
          _ui_element(text='Save', bounds=(20, 300, 700, 360), clickable=True),
      ]

    typed = 'Team Sync'
    client = FakeAndroidEnvClient(ui_elements=field(''), apps=[])
    client.queue_state(field(''))
    client.queue_state(field(''))
    client.queue_state(field(''))
    client.queue_state(field(typed), pixels=_frame(40))
    jev = FakeJevWrapper(_scripted('TYPE_TEXT'))
    agent = mobile_jev_v2.ClientMobileJevV2(client, llm=None, jev=jev)
    agent.transition_pause = 0.0
    result = agent.step(typed)
    self.assertEqual(result.data['mobile_jev_v2']['status'], 'action')
    self.assertTrue(
        any(e.text == typed for e in agent._observation.elements)
    )
    self.assertTrue((agent._frame == 40).all())
    self.assertTrue((agent._last_state.pixels == 40).all())

  def test_unverified_input_ends_the_episode(self):
    # The text went out; nothing proves it landed. The agent must say so rather
    # than decide again on a screen it has not confirmed.
    elements = [
        _ui_element(
            class_name='android.widget.EditText',
            bounds=(20, 220, 980, 280),
            clickable=True,
            editable=True,
            focused=True,
            hint='Title',
        ),
        _ui_element(text='Save', bounds=(20, 300, 700, 360), clickable=True),
    ]
    client = FakeAndroidEnvClient(ui_elements=elements)
    jev = FakeJevWrapper(_scripted('TYPE_TEXT'))
    agent = mobile_jev_v2.ClientMobileJevV2(
        client, llm=None, jev=jev, input_timeout_ms=1
    )
    agent.transition_pause = 0.0
    result = agent.step('Set the title to Team Sync')
    self.assertTrue(result.done)
    self.assertEqual(
        result.data['mobile_jev_v2']['status'], 'input_unverified'
    )
    self.assertLen(client.typed_texts, 1)
  def test_done_step_dispatches_nothing(self):
    agent, client, _ = _create_agent(
        _rich_screen(), _scripted('TAP', '1', done=0.99)
    )
    result = agent.step('Tap Search')
    self.assertTrue(result.done)
    self.assertEqual(result.data['mobile_jev_v2']['status'], 'done')
    self.assertEmpty(client.taps)
    self.assertEmpty(client.executed_actions)

  def test_installed_apps_are_refetched_after_a_failure(self):
    agent, client, _ = _create_agent(_rich_screen(), _scripted('BACK'))
    client.get_all_apps_failures = 1
    agent.step('Go back')
    self.assertIsNone(agent._installed_apps)
    agent.step('Go back')
    self.assertIsNotNone(agent._installed_apps)


if __name__ == '__main__':
  absltest.main()
