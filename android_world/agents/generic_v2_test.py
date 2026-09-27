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

"""Tests for the self-contained GenericAgentV2 port."""

import json
import os
from unittest import mock

from absl.testing import absltest
from android_world import constants
from android_world.agents import generic_v2
from android_world.env import interface
from android_world.env import json_action
import numpy as np


_SCREEN = (1080, 2400)


class FakeAndroidEnvClient:
  """Duck-typed stand-in for interface.AndroidEnvClient.

  Implements only the surface ClientGenericR2SOL uses, so tests never touch
  HTTP.
  """

  def __init__(self, pixels: np.ndarray | None = None):
    self.reset_go_home = None
    self.hide_automation_ui_called = False
    self.executed_actions = []
    self.pressed_keys = []
    self.typed_texts = []
    self.generic_requests = []
    if pixels is None:
      pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    self.pixels = pixels

  def reset(self, go_home: bool) -> None:
    self.reset_go_home = go_home

  def hide_automation_ui(self) -> None:
    self.hide_automation_ui_called = True

  def get_state(self, wait_to_stabilize: bool = False) -> interface.State:
    del wait_to_stabilize
    return interface.State(
        pixels=self.pixels,
        forest=None,
        ui_elements=[],
        auxiliaries={},
    )

  def get_logical_screen_size(self) -> tuple[int, int]:
    return _SCREEN

  def execute_action(self, action: json_action.JSONAction) -> None:
    self.executed_actions.append(action)

  def press_key(self, keycode: str, timeout_sec: float = 10) -> None:
    del timeout_sec
    self.pressed_keys.append(keycode)

  def type_text(self, text: str, timeout_sec: float = 10) -> None:
    del timeout_sec
    self.typed_texts.append(text)

  def issue_generic_request(self, args, timeout_sec: float = 10) -> dict:
    del timeout_sec
    self.generic_requests.append(args)
    return {}


def _create_agent(
    responses: list[str],
    client: FakeAndroidEnvClient | None = None,
):
  """Builds an agent whose model calls return `responses` in order."""
  client = client or FakeAndroidEnvClient()
  with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
    agent = generic_v2.ClientGenericR2SOL(
        client, base_url='http://localhost:1/v1', model_name='fake'
    )
  agent._call_model = mock.MagicMock(side_effect=list(responses))
  return agent, client


class ResponseParsingTest(absltest.TestCase):

  def test_think_and_answer_are_extracted_case_insensitively(self):
    action = generic_v2.parse_response(
        '<think>Looking around.</think>'
        '<ANSWER>{"action":"tap","point":[20,30]}</ANSWER>'
    )

    self.assertEqual(action.action, generic_v2.ActionName.CLICK)
    self.assertEqual(action.thought, 'Looking around.')
    self.assertEqual(action.payload['point'], [20, 30])

  def test_action_aliases_are_normalized(self):
    aliases = {
        'TAP': 'CLICK',
        'DOUBLETAP': 'DOUBLE_TAP',
        'LONG_PRESS': 'LONGPRESS',
        'SLIDE': 'SWIPE',
        'LAUNCH': 'AWAKE',
        'FINISH': 'COMPLETE',
    }
    for alias, expected in aliases.items():
      with self.subTest(alias=alias):
        response = f'<answer>{{"action":"{alias}"}}</answer>'
        action = generic_v2.parse_response(response)
        self.assertEqual(action.action.value, expected)

  def test_unparseable_response_fails_closed_with_a_stable_code(self):
    cases = [
        ('', 'empty_response'),
        ('not JSON', 'invalid_json'),
        ('<answer>{"action":"BACK"</answer>', 'invalid_json'),
        ('<answer>{"action":"RUN_SHELL"}</answer>', 'unknown_action'),
        ('<answer>["BACK"]</answer>', 'invalid_json'),
    ]
    for response, code in cases:
      with self.subTest(response=response):
        action = generic_v2.parse_response(response)
        self.assertEqual(action.action, generic_v2.ActionName.ABORT)
        self.assertEqual(action.payload, {'action': 'ABORT', 'value': code})

  def test_coordinates_are_clamped_to_the_normalized_range(self):
    action = generic_v2.parse_response(
        '<answer>{"action":"CLICK","point":[1500,-20]}</answer>'
    )

    self.assertEqual(action.payload['point'], [1000, 0])

  def test_non_pointer_actions_are_not_clamped(self):
    action = generic_v2.parse_response(
        '<answer>{"action":"SWIPE","point1":[1200,500],'
        '"point2":[-5,5]}</answer>'
    )

    self.assertEqual(action.payload['point1'], [1000, 500])
    self.assertEqual(action.payload['point2'], [0, 5])

  def test_response_wrapped_in_prose_is_recovered(self):
    action = generic_v2.parse_response(
        'Sure. <answer>{"action":"CLICK","point":[10,10]}</answer> done.'
    )

    self.assertEqual(action.action, generic_v2.ActionName.CLICK)


class ActionTranslationTest(absltest.TestCase):

  def _translate(self, response: str):
    return generic_v2._to_json_action(
        generic_v2.parse_response(response), _SCREEN
    )

  def test_pointer_actions_scale_into_pixels(self):
    cases = {
        'CLICK': json_action.CLICK,
        'DOUBLE_TAP': json_action.DOUBLE_TAP,
        'LONGPRESS': json_action.LONG_PRESS,
    }
    for name, action_type in cases.items():
      with self.subTest(action=name):
        converted = self._translate(
            f'<answer>{{"action":"{name}","point":[500,500]}}</answer>'
        )
        self.assertEqual(converted.action_type, action_type)
        self.assertEqual((converted.x, converted.y), (540, 1200))

  def test_coordinates_are_clamped_before_scaling(self):
    converted = self._translate(
        '<answer>{"action":"CLICK","point":[2000,2000]}</answer>'
    )

    self.assertEqual((converted.x, converted.y), (1080, 2400))

  def test_type_returns_only_the_focus_click(self):
    # Typing must not go through `input_text`, which always presses ENTER.
    converted = self._translate(
        '<answer>{"action":"TYPE","value":"hi","point":[0,0]}</answer>'
    )

    self.assertEqual(converted.action_type, json_action.CLICK)
    self.assertIsNone(converted.text)
    self.assertIsNone(converted.clear_text)

  def test_swipe_is_a_fling_not_a_long_press_drag(self):
    # `drag_and_drop` is a 4000ms hold, which would long-press a list item
    # instead of scrolling it.
    converted = self._translate(
        '<answer>{"action":"SWIPE","point1":[100,900],"point2":[100,100]}'
        '</answer>'
    )

    self.assertEqual(converted.action_type, json_action.SWIPE)
    self.assertEqual(converted.direction, 'up')

  def test_swipe_direction_follows_the_dominant_axis(self):
    cases = [
        ((500, 900), (500, 100), 'up'),
        ((500, 100), (500, 900), 'down'),
        ((100, 500), (900, 500), 'right'),
        ((900, 500), (100, 500), 'left'),
    ]
    for start, end, expected in cases:
      with self.subTest(expected=expected):
        converted = self._translate(
            f'<answer>{{"action":"SWIPE","point1":{list(start)},'
            f'"point2":{list(end)}}}</answer>'
        )
        self.assertEqual(converted.direction, expected)

  def test_swipe_accepts_a_bare_direction(self):
    converted = self._translate(
        '<answer>{"action":"SWIPE","direction":"up"}</answer>'
    )

    self.assertEqual(converted.action_type, json_action.SWIPE)
    self.assertEqual(converted.direction, 'up')

  def test_drag_preserves_exact_points(self):
    converted = self._translate(
        '<answer>{"action":"DRAG","point1":[100,900],"point2":[100,100]}'
        '</answer>'
    )

    self.assertEqual(converted.action_type, json_action.DRAG_AND_DROP)
    self.assertEqual(converted.touch_xy, [108, 2160])
    self.assertEqual(converted.lift_xy, [108, 240])

  def test_degenerate_swipe_falls_back_to_a_drag(self):
    # No direction is derivable, so honor the exact points instead of guessing.
    converted = self._translate(
        '<answer>{"action":"SWIPE","point1":[500,500],"point2":[500,500]}'
        '</answer>'
    )

    self.assertEqual(converted.action_type, json_action.DRAG_AND_DROP)

  def test_simple_actions_map_to_their_counterparts(self):
    cases = [
        ('BACK', json_action.NAVIGATE_BACK),
        ('HOME', json_action.NAVIGATE_HOME),
        ('ENTER', json_action.KEYBOARD_ENTER),
        ('WAIT', json_action.WAIT),
    ]
    for name, action_type in cases:
      with self.subTest(action=name):
        converted = self._translate(f'<answer>{{"action":"{name}"}}</answer>')
        self.assertEqual(converted.action_type, action_type)

  def test_awake_maps_to_open_app(self):
    converted = self._translate(
        '<answer>{"action":"AWAKE","value":"Settings"}</answer>'
    )

    self.assertEqual(converted.action_type, json_action.OPEN_APP)
    self.assertEqual(converted.app_name, 'Settings')

  def test_answer_carries_its_text(self):
    converted = self._translate(
        '<answer>{"action":"ANSWER","value":"42"}</answer>'
    )

    self.assertEqual(converted.action_type, json_action.ANSWER)
    self.assertEqual(converted.text, '42')

  def test_terminal_actions_map_to_goal_status(self):
    complete = self._translate('<answer>{"action":"COMPLETE"}</answer>')
    self.assertEqual(complete.action_type, json_action.STATUS)
    self.assertEqual(complete.goal_status, 'complete')

    abort = self._translate(
        '<answer>{"action":"ABORT","value":"stuck"}</answer>'
    )
    self.assertEqual(abort.action_type, json_action.STATUS)
    self.assertEqual(abort.goal_status, 'infeasible')

  def test_recent_has_no_json_action_equivalent(self):
    self.assertIsNone(self._translate('<answer>{"action":"RECENT"}</answer>'))

  def test_missing_required_fields_raise(self):
    cases = [
        '<answer>{"action":"CLICK"}</answer>',
        '<answer>{"action":"TYPE","point":[10,10]}</answer>',
        '<answer>{"action":"AWAKE"}</answer>',
    ]
    for response in cases:
      with self.subTest(response=response):
        with self.assertRaises(ValueError):
          self._translate(response)


class AbortClassificationTest(absltest.TestCase):

  def test_parse_failures_are_recoverable(self):
    for response in ['', 'not JSON', '<answer>{"action":"RUN"}</answer>']:
      with self.subTest(response=response):
        action = generic_v2.parse_response(response)
        self.assertTrue(generic_v2._is_recoverable_abort(action))

  def test_a_deliberate_model_abort_is_terminal(self):
    action = generic_v2.parse_response(
        '<answer>{"action":"ABORT","value":"no such contact"}</answer>'
    )

    self.assertFalse(generic_v2._is_recoverable_abort(action))

  def test_completion_is_not_an_abort(self):
    self.assertFalse(
        generic_v2._is_recoverable_abort(
            generic_v2.parse_response('<answer>{"action":"COMPLETE"}</answer>')
        )
    )


class CoreTest(absltest.TestCase):

  _IMAGE = 'data:image/jpeg;base64,AAAA'

  def _core(self) -> generic_v2.GenericV2Core:
    core = generic_v2.GenericV2Core()
    core.reset('open settings')
    return core

  def test_build_messages_before_reset_raises(self):
    with self.assertRaises(RuntimeError):
      generic_v2.GenericV2Core().build_messages(0, self._IMAGE)

  def test_history_is_bounded_to_two_turns(self):
    core = self._core()
    for step in range(5):
      core.build_messages(step, self._IMAGE)
      core.remember_observation_image(self._IMAGE)
      core.interpret('<answer>{"action":"BACK"}</answer>')

    self.assertLen(core.history, 2)

  def test_fact_ledger_is_capped(self):
    core = self._core()
    facts = {f'k{i}': f'v{i}' for i in range(40)}
    core.interpret(
        f'<answer>{{"action":"BACK"}}</answer>'
        f'<facts>{json.dumps(facts)}</facts>'
    )

    self.assertLen(core.fact_ledger, 24)

  def test_oversized_facts_are_rejected(self):
    core = self._core()
    core.interpret(
        f'<answer>{{"action":"BACK"}}</answer>'
        f'<facts>{json.dumps({"k": "v" * 501, "good": "value"})}</facts>'
    )

    self.assertEqual(core.fact_ledger, {'good': 'value'})

  def test_no_progress_escape_rewrites_a_click_to_back(self):
    core = self._core()
    click = '<answer>{"action":"CLICK","point":[100,100]}</answer>'
    for step in range(3):
      core.build_messages(step, self._IMAGE)
      core.remember_observation_image(self._IMAGE)
      served = core.interpret(click)

    self.assertTrue(core.route_replan_required)
    self.assertEqual(served.action, generic_v2.ActionName.BACK)

  def test_escape_that_still_shows_no_progress_aborts(self):
    core = self._core()
    click = '<answer>{"action":"CLICK","point":[100,100]}</answer>'
    for step in range(3):
      core.build_messages(step, self._IMAGE)
      core.remember_observation_image(self._IMAGE)
      core.interpret(click)
    # The model ignores the BACK constraint and goes back to clicking.
    core.build_messages(3, self._IMAGE)
    core.remember_observation_image(self._IMAGE)
    served = core.interpret(click)

    self.assertEqual(served.action, generic_v2.ActionName.ABORT)
    self.assertEqual(served.payload['value'], 'route_replan_no_progress')
    self.assertFalse(generic_v2._is_recoverable_abort(served))

  def test_visible_progress_clears_the_replan_block(self):
    core = self._core()
    click = '<answer>{"action":"CLICK","point":[100,100]}</answer>'
    for step in range(3):
      core.build_messages(step, self._IMAGE)
      core.remember_observation_image(self._IMAGE)
      core.interpret(click)

    core.build_messages(3, self._IMAGE + '-changed')
    self.assertFalse(core.route_replan_required)

  def test_state_changes_are_compared_against_the_previous_image(self):
    core = self._core()
    core.build_messages(0, self._IMAGE)
    core.remember_observation_image(self._IMAGE)
    core.interpret('<answer>{"action":"BACK"}</answer>')

    messages = core.build_messages(1, self._IMAGE)
    texts = [
        item['text']
        for item in messages[-1]['content']
        if item['type'] == 'text'
    ]

    self.assertIn('[动作效果核验]', '\n'.join(texts))
    self.assertIn('无效动作警告', '\n'.join(texts))


class ClientGenericR2SOLTest(absltest.TestCase):

  def test_reset_hides_automation_ui_and_clears_history(self):
    agent, client = _create_agent([])
    agent.reset(go_home_on_reset=True)

    self.assertTrue(client.reset_go_home)
    self.assertTrue(client.hide_automation_ui_called)
    self.assertEmpty(agent.history)

  def test_missing_api_key_raises(self):
    environment = {k: v for k, v in os.environ.items() if k != 'OPENAI_API_KEY'}
    with mock.patch.dict(os.environ, environment, clear=True):
      with self.assertRaises(RuntimeError):
        generic_v2.ClientGenericR2SOL(
            FakeAndroidEnvClient(),
            base_url='http://localhost:1/v1',
            model_name='m',
        )

  def test_completion_terminates_without_executing_an_action(self):
    agent, client = _create_agent(['<answer>{"action":"COMPLETE"}</answer>'])

    result = agent.step('do something')

    self.assertTrue(result.done)
    self.assertEmpty(client.executed_actions)

  def test_a_deliberate_abort_terminates(self):
    agent, _ = _create_agent(
        ['<answer>{"action":"ABORT","value":"no such contact"}</answer>']
    )

    result = agent.step('do something')

    self.assertTrue(result.done)

  def test_unparseable_response_does_not_terminate_the_episode(self):
    agent, client = _create_agent(
        ['garbage', '<answer>{"action":"COMPLETE"}</answer>']
    )

    first = agent.step('do something')
    second = agent.step('do something')

    self.assertFalse(first.done)
    self.assertTrue(second.done)
    self.assertEmpty(client.executed_actions)

  def test_click_is_scaled_and_executed(self):
    agent, client = _create_agent(
        ['<answer>{"action":"CLICK","point":[500,500]}</answer>']
    )

    agent.step('do something')

    self.assertLen(client.executed_actions, 1)
    self.assertEqual(client.executed_actions[0].action_type, json_action.CLICK)
    self.assertEqual(
        (client.executed_actions[0].x, client.executed_actions[0].y),
        (540, 1200),
    )

  def test_recent_uses_the_recents_key_instead_of_an_action(self):
    agent, client = _create_agent(['<answer>{"action":"RECENT"}</answer>'])

    agent.step('do something')

    self.assertEqual(client.pressed_keys, ['KEYCODE_APP_SWITCH'])
    self.assertEmpty(client.executed_actions)

  def test_type_focuses_then_types_without_submitting(self):
    agent, client = _create_agent(
        ['<answer>{"action":"TYPE","value":"hello","point":[0,0]}</answer>']
    )

    agent.step('do something')

    self.assertLen(client.executed_actions, 1)
    self.assertEqual(client.executed_actions[0].action_type, json_action.CLICK)
    self.assertEqual(client.typed_texts, ['hello'])
    # `input_text` would have pressed ENTER; assert we never used it.
    self.assertNotIn(
        json_action.INPUT_TEXT,
        [action.action_type for action in client.executed_actions],
    )

  def test_type_with_clear_selects_all_then_deletes(self):
    agent, client = _create_agent([
        '<answer>{"action":"TYPE","value":"x","point":[0,0],'
        '"clear":true}</answer>'
    ])

    agent.step('do something')

    self.assertIn(
        ['shell', 'input', 'keycombination', '113', '29'],
        client.generic_requests,
    )
    self.assertIn('KEYCODE_DEL', client.pressed_keys)

  def test_a_null_content_response_fails_closed_instead_of_crashing(self):
    # vLLM reports a null `content` for a reasoning-only turn. The None used to
    # reach `GenericV2Core.interpret` -> `_remember_facts`, where re.search
    # raised and took the whole task down as a SKIPPED episode.
    client = FakeAndroidEnvClient()
    with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
      agent = generic_v2.ClientGenericR2SOL(
          client, base_url='http://localhost:1/v1', model_name='m'
      )
    agent.client_sdk = mock.MagicMock()
    message = mock.MagicMock()
    message.content = None
    agent.client_sdk.chat.completions.create.return_value = mock.MagicMock(
        choices=[message]
    )

    result = agent.step('do something')

    self.assertFalse(result.done)
    self.assertEqual(result.data['action_output'], '')
    self.assertEmpty(client.executed_actions)

  def test_execution_failure_is_not_terminal(self):
    agent, client = _create_agent(
        [
            '<answer>{"action":"CLICK","point":[500,500]}</answer>',
            '<answer>{"action":"COMPLETE"}</answer>',
        ]
    )
    client.execute_action = mock.MagicMock(side_effect=RuntimeError('HTTP 500'))

    first = agent.step('do something')
    second = agent.step('do something')

    self.assertFalse(first.done)
    self.assertTrue(second.done)

  def test_step_data_has_the_expected_keys_and_no_step_number(self):
    agent, _ = _create_agent(['<answer>{"action":"BACK"}</answer>'])

    result = agent.step('do something')

    expected = {
        'before_screenshot',
        'after_screenshot',
        'before_element_list',
        'after_element_list',
        'action_prompt',
        'action_output',
        'action_raw_response',
        'summary_prompt',
        'summary',
        'summary_raw_response',
    }
    self.assertEqual(set(result.data), expected)
    self.assertNotIn(constants.STEP_NUMBER, result.data)

  def test_step_data_never_embeds_the_screenshot_data_url(self):
    agent, _ = _create_agent(['<answer>{"action":"BACK"}</answer>'])

    result = agent.step('do something')

    # Screenshots live in the *_screenshot keys as arrays; the textual prompt
    # must not carry the base64 payload into the episode log.
    self.assertNotIn('base64', result.data['action_prompt'])

  def test_model_call_receives_the_upstream_sampling_parameters(self):
    agent, _ = _create_agent(['<answer>{"action":"BACK"}</answer>'])

    agent.step('do something')

    call_args = agent._call_model.call_args
    model_args = call_args[0][1]
    # The point is that the call carries the upstream budget, which
    # `infer.OpenAIWrapper` cannot express: it hardcodes 1000.
    self.assertGreater(model_args['max_tokens'], 1000)
    self.assertEqual(model_args['temperature'], 0.1)

  def test_an_empty_goal_does_not_crash_the_episode(self):
    agent, client = _create_agent([])

    result = agent.step('   ')

    self.assertFalse(result.done)
    self.assertEmpty(client.executed_actions)
    agent._call_model.assert_not_called()

  def test_the_core_task_is_established_from_the_first_step(self):
    agent, _ = _create_agent(
        [
            '<answer>{"action":"BACK"}</answer>',
            '<answer>{"action":"BACK"}</answer>',
        ]
    )

    agent.step('first goal')
    self.assertEqual(agent.core.task, 'first goal')
    self.assertEqual(agent._step_index, 1)

    # A new goal resets the core, so the step counter restarts with it.
    agent.step('a different goal')
    self.assertEqual(agent.core.task, 'a different goal')
    self.assertEqual(agent._step_index, 1)


def _create_generic_agent(
    responses: list[str],
    client: FakeAndroidEnvClient | None = None,
):
  """Builds a baseline ClientGeneric whose model calls return `responses`."""
  client = client or FakeAndroidEnvClient()
  with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
    agent = generic_v2.ClientGeneric(
        client, base_url='http://localhost:1/v1', model_name='fake'
    )
  agent._call_model = mock.MagicMock(side_effect=list(responses))
  return agent, client


class ClientGenericTest(absltest.TestCase):

  def test_the_prompt_is_the_baseline_not_the_r2sol_one(self):
    prompt = generic_v2.ClientGeneric.SYSTEM_PROMPT

    self.assertIn('<THINK>', prompt)
    self.assertIn('<ANSWER>', prompt)
    # The R2-SOL gates and the fact ledger must not leak into the baseline.
    self.assertNotIn('FACTS', prompt)

  def test_model_args_override_the_upstream_defaults(self):
    with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
      agent = generic_v2.ClientGeneric(
          FakeAndroidEnvClient(),
          base_url='http://localhost:1/v1',
          model_name='m',
          model_args={'temperature': 0.7},
      )

    self.assertEqual(agent.model_args['temperature'], 0.7)
    self.assertEqual(
        agent.model_args['max_tokens'],
        generic_v2.ClientGeneric.DEFAULT_MODEL_ARGS['max_tokens'],
    )

  def test_reset_hides_automation_ui_and_clears_history(self):
    agent, client = _create_generic_agent(
        ['<answer>{"action":"BACK"}</answer>']
    )
    agent.step('do something')
    self.assertLen(agent.history, 1)

    agent.reset(go_home_on_reset=True)

    self.assertTrue(client.reset_go_home)
    self.assertTrue(client.hide_automation_ui_called)
    self.assertEmpty(agent.history)

  def test_missing_api_key_raises(self):
    environment = {k: v for k, v in os.environ.items() if k != 'OPENAI_API_KEY'}
    with mock.patch.dict(os.environ, environment, clear=True):
      with self.assertRaises(RuntimeError):
        generic_v2.ClientGeneric(
            FakeAndroidEnvClient(),
            base_url='http://localhost:1/v1',
            model_name='m',
        )

  def test_completion_terminates_without_executing_an_action(self):
    agent, client = _create_generic_agent(
        ['<answer>{"action":"COMPLETE"}</answer>']
    )

    result = agent.step('do something')

    self.assertTrue(result.done)
    self.assertEmpty(client.executed_actions)

  def test_a_deliberate_abort_terminates(self):
    agent, _ = _create_generic_agent(
        ['<answer>{"action":"ABORT","value":"no such contact"}</answer>']
    )

    result = agent.step('do something')

    self.assertTrue(result.done)

  def test_an_answer_is_recorded_without_ending_the_episode(self):
    # Unlike ClientGenericR2SOL, the baseline only ends on COMPLETE or ABORT.
    agent, client = _create_generic_agent([
        '<answer>{"action":"ANSWER","value":"42"}</answer>',
        '<answer>{"action":"COMPLETE"}</answer>',
    ])

    first = agent.step('do something')
    second = agent.step('do something')

    self.assertFalse(first.done)
    self.assertTrue(second.done)
    self.assertLen(client.executed_actions, 1)
    self.assertEqual(client.executed_actions[0].action_type, json_action.ANSWER)
    self.assertEqual(client.executed_actions[0].text, '42')

  def test_unparseable_response_does_not_terminate_the_episode(self):
    agent, client = _create_generic_agent(
        ['garbage', '<answer>{"action":"COMPLETE"}</answer>']
    )

    first = agent.step('do something')
    second = agent.step('do something')

    self.assertFalse(first.done)
    self.assertTrue(second.done)
    self.assertEmpty(client.executed_actions)

  def test_a_null_content_response_fails_closed_instead_of_crashing(self):
    # vLLM reports a null `content` for a reasoning-only turn, which reached
    # `print('Response: ' + raw_response)` and killed the whole task.
    client = FakeAndroidEnvClient()
    with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
      agent = generic_v2.ClientGeneric(
          client, base_url='http://localhost:1/v1', model_name='m'
      )
    agent.client_sdk = mock.MagicMock()
    message = mock.MagicMock()
    message.content = None
    agent.client_sdk.chat.completions.create.return_value = mock.MagicMock(
        choices=[message]
    )

    result = agent.step('do something')

    self.assertFalse(result.done)
    self.assertEqual(result.data['action_output'], '')
    self.assertEmpty(client.executed_actions)

  def test_execution_failure_is_not_terminal(self):
    agent, client = _create_generic_agent(
        [
            '<answer>{"action":"CLICK","point":[500,500]}</answer>',
            '<answer>{"action":"COMPLETE"}</answer>',
        ]
    )
    client.execute_action = mock.MagicMock(side_effect=RuntimeError('HTTP 500'))

    first = agent.step('do something')
    second = agent.step('do something')

    self.assertFalse(first.done)
    self.assertTrue(second.done)

  def test_click_is_scaled_and_executed(self):
    agent, client = _create_generic_agent(
        ['<answer>{"action":"CLICK","point":[500,500]}</answer>']
    )

    agent.step('do something')

    self.assertLen(client.executed_actions, 1)
    self.assertEqual(client.executed_actions[0].action_type, json_action.CLICK)
    self.assertEqual(
        (client.executed_actions[0].x, client.executed_actions[0].y),
        (540, 1200),
    )

  def test_recent_uses_the_recents_key_instead_of_an_action(self):
    agent, client = _create_generic_agent(
        ['<answer>{"action":"RECENT"}</answer>']
    )

    agent.step('do something')

    self.assertEqual(client.pressed_keys, ['KEYCODE_APP_SWITCH'])
    self.assertEmpty(client.executed_actions)

  def test_type_without_a_point_types_into_the_focused_field(self):
    # The upstream prompt marks `point` optional for TYPE.
    agent, client = _create_generic_agent(
        ['<answer>{"action":"TYPE","value":"hi"}</answer>']
    )

    agent.step('do something')

    self.assertEmpty(client.executed_actions)
    self.assertEqual(client.typed_texts, ['hi'])

  def test_type_with_a_point_focuses_then_types_without_submitting(self):
    agent, client = _create_generic_agent(
        [
            '<answer>{"action":"TYPE","text":"hi","point":[0,0]}</answer>'
        ]
    )

    agent.step('do something')

    self.assertLen(client.executed_actions, 1)
    self.assertEqual(client.executed_actions[0].action_type, json_action.CLICK)
    self.assertEqual(client.typed_texts, ['hi'])
    # `input_text` would have pressed ENTER; assert we never used it.
    self.assertNotIn(
        json_action.INPUT_TEXT,
        [action.action_type for action in client.executed_actions],
    )

  def test_type_with_clear_selects_all_then_deletes(self):
    agent, client = _create_generic_agent([
        '<answer>{"action":"TYPE","value":"x","point":[0,0],'
        '"clear":true}</answer>'
    ])

    agent.step('do something')

    self.assertIn(
        ['shell', 'input', 'keycombination', '113', '29'],
        client.generic_requests,
    )
    self.assertIn('KEYCODE_DEL', client.pressed_keys)

  def test_swipe_uses_the_upstream_start_and_end_points(self):
    agent, client = _create_generic_agent(
        [
            '<answer>{"action":"SWIPE","start":[100,900],"end":[100,100]}'
            '</answer>'
        ]
    )

    agent.step('do something')

    self.assertLen(client.executed_actions, 1)
    self.assertEqual(client.executed_actions[0].action_type, json_action.SWIPE)
    self.assertEqual(client.executed_actions[0].direction, 'up')

  def test_awake_accepts_the_upstream_app_alias(self):
    agent, client = _create_generic_agent(
        ['<answer>{"action":"AWAKE","app":"Clock"}</answer>']
    )

    agent.step('do something')

    self.assertEqual(
        client.executed_actions[0].action_type, json_action.OPEN_APP
    )
    self.assertEqual(client.executed_actions[0].app_name, 'Clock')

  def test_wait_defaults_to_one_second_and_accepts_duration(self):
    agent, client = _create_generic_agent([
        '<answer>{"action":"WAIT"}</answer>',
        '<answer>{"action":"WAIT","duration":3}</answer>',
    ])
    client.execute_action = mock.MagicMock()

    with mock.patch.object(generic_v2.time, 'sleep') as sleep:
      agent.step('do something')
      sleep.assert_any_call(1.0)
      agent.step('do something')
      sleep.assert_any_call(3.0)

  def test_payload_aliases_fill_the_canonical_keys(self):
    agent, _ = _create_generic_agent([])
    cases = [
        ('{"action":"TYPE","text":"hi"}', 'value', 'hi'),
        ('{"action":"SWIPE","start":[1,2],"end":[3,4]}', 'point1', [1, 2]),
        ('{"action":"SWIPE","start":[1,2],"end":[3,4]}', 'point2', [3, 4]),
        ('{"action":"WAIT","duration":3}', 'value', 3),
        ('{"action":"AWAKE","app":"Clock"}', 'value', 'Clock'),
        ('{"action":"ANSWER","text":"42"}', 'value', '42'),
        ('{"action":"COMPLETE","message":"done"}', 'return', 'done'),
        ('{"action":"ABORT","reason":"stuck"}', 'value', 'stuck'),
    ]
    for response, key, expected in cases:
      with self.subTest(response=response):
        normalized = agent._normalize_payload(
            generic_v2.parse_response(response)
        )
        self.assertEqual(normalized.payload[key], expected)

  def test_the_full_text_history_is_replayed(self):
    agent, _ = _create_generic_agent([
        '<answer>{"action":"BACK"}</answer>',
        '<answer>{"action":"HOME"}</answer>',
        '<answer>{"action":"ENTER"}</answer>',
    ])
    for _ in range(3):
      agent.step('do something')

    messages = agent._build_messages('data:image/jpeg;base64,x')

    # System, three replayed turns, and the current screenshot turn.
    self.assertLen(messages, 8)
    self.assertEqual(messages[0]['role'], 'system')
    self.assertIn('[任务]', messages[1]['content'][0]['text'])
    self.assertEqual(
        messages[2]['content'], '<answer>{"action":"BACK"}</answer>'
    )
    self.assertEqual(messages[3]['content'][0]['text'], '[Step 2]')
    self.assertEqual(
        messages[4]['content'], '<answer>{"action":"HOME"}</answer>'
    )
    self.assertEqual(messages[5]['content'][0]['text'], '[Step 3]')
    self.assertEqual(messages[7]['content'][1]['text'], '[Step 4]')

  def test_a_new_goal_clears_the_replayed_history(self):
    agent, _ = _create_generic_agent([
        '<answer>{"action":"BACK"}</answer>',
        '<answer>{"action":"HOME"}</answer>',
    ])
    agent.step('first goal')
    agent.step('second goal')

    messages = agent._build_messages('data:image/jpeg;base64,x')

    self.assertLen(messages, 4)
    self.assertIn('second goal', messages[1]['content'][0]['text'])

  def test_step_data_has_the_expected_keys_and_no_step_number(self):
    agent, _ = _create_generic_agent(['<answer>{"action":"BACK"}</answer>'])

    result = agent.step('do something')

    expected = {
        'before_screenshot',
        'after_screenshot',
        'before_element_list',
        'after_element_list',
        'action_prompt',
        'action_output',
        'action_raw_response',
        'summary_prompt',
        'summary',
        'summary_raw_response',
    }
    self.assertEqual(set(result.data), expected)
    self.assertNotIn(constants.STEP_NUMBER, result.data)

  def test_step_data_never_embeds_the_screenshot_data_url(self):
    agent, _ = _create_generic_agent(['<answer>{"action":"BACK"}</answer>'])

    result = agent.step('do something')

    self.assertNotIn('base64', result.data['action_prompt'])

  def test_model_call_receives_the_upstream_sampling_parameters(self):
    agent, _ = _create_generic_agent(['<answer>{"action":"BACK"}</answer>'])

    agent.step('do something')

    model_args = agent._call_model.call_args[0][1]
    # Read the configured budget rather than pinning the number, so raising it
    # does not turn this into a red test.
    self.assertEqual(
        model_args['max_tokens'],
        generic_v2.ClientGeneric.DEFAULT_MODEL_ARGS['max_tokens'],
    )
    self.assertEqual(model_args['temperature'], 0.1)
    self.assertEqual(model_args['top_p'], 0.95)

  def test_an_empty_goal_does_not_crash_the_episode(self):
    agent, client = _create_generic_agent([])

    result = agent.step('   ')

    self.assertFalse(result.done)
    self.assertEmpty(client.executed_actions)
    agent._call_model.assert_not_called()


class ModelTransportTest(absltest.TestCase):
  """Covers the `_call_model` transport, mirroring infer.OpenAIWrapper."""

  def _agent(self, base_url: str, **kwargs):
    with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
      return generic_v2.ClientGenericR2SOL(
          FakeAndroidEnvClient(), base_url=base_url, model_name='m', **kwargs
      )

  def test_chat_url_tolerates_a_trailing_v1(self):
    # `--base_url` is conventionally given with a trailing /v1, which the openai
    # SDK wants but which must not be doubled up for the requests path.
    cases = [
        ('http://host:1234', 'http://host:1234/v1/chat/completions'),
        ('http://host:1234/', 'http://host:1234/v1/chat/completions'),
        ('http://host:1234/v1', 'http://host:1234/v1/chat/completions'),
        ('http://host:1234/v1/', 'http://host:1234/v1/chat/completions'),
    ]
    for base, expected in cases:
      with self.subTest(base=base):
        self.assertEqual(self._agent(base)._chat_url(), expected)

  def test_max_retry_is_clamped_like_the_wrapper(self):
    self.assertEqual(self._agent('http://host', max_retry=99).max_retry, 5)
    self.assertEqual(self._agent('http://host', max_retry=0).max_retry, 3)
    self.assertEqual(self._agent('http://host', max_retry=3).max_retry, 3)

  def _ok_response(self, content: str):
    response = mock.MagicMock(ok=True, status_code=200)
    response.json.return_value = {
        'choices': [{'message': {'content': content}}]
    }
    return response

  def test_requests_fallback_posts_the_normalized_url_and_sampling_args(self):
    agent = self._agent('https://router.example/v1')
    agent.client_sdk = None
    response = self._ok_response('hello')

    with mock.patch.object(
        generic_v2.requests, 'post', return_value=response
    ) as post:
      result = agent._call_model(
          [{'role': 'user', 'content': 'hi'}], {'max_tokens': 8192}
      )

    self.assertEqual(result, 'hello')
    self.assertEqual(
        post.call_args[0][0], 'https://router.example/v1/chat/completions'
    )
    self.assertEqual(post.call_args[1]['json']['max_tokens'], 8192)

  def test_sdk_path_is_preferred_and_honors_sampling_args(self):
    agent = self._agent('http://host')
    agent.client_sdk = mock.MagicMock()
    choice = mock.MagicMock()
    choice.message.content = 'sdk-hello'
    agent.client_sdk.chat.completions.create.return_value = mock.MagicMock(
        choices=[choice]
    )

    result = agent._call_model(
        [{'role': 'user', 'content': 'x'}], {'temperature': 0.1}
    )

    self.assertEqual(result, 'sdk-hello')
    kwargs = agent.client_sdk.chat.completions.create.call_args[1]
    self.assertEqual(kwargs['model'], 'm')
    self.assertEqual(kwargs['temperature'], 0.1)

  def test_retries_are_bounded_and_return_the_sentinel(self):
    # OpenAIWrapper only decrements its counter on exceptions, so an endpoint
    # that keeps answering with an error body spins forever. Every failed
    # attempt must consume a retry.
    agent = self._agent('http://host', max_retry=3)
    agent.client_sdk = None
    response = mock.MagicMock(ok=False, status_code=500, text='boom')

    with mock.patch.object(
        generic_v2.requests, 'post', return_value=response
    ) as post:
      with mock.patch.object(generic_v2.time, 'sleep'):
        result = agent._call_model([{'role': 'user', 'content': 'x'}], {})

    self.assertEqual(result, generic_v2.infer.ERROR_CALLING_LLM)
    self.assertEqual(post.call_count, 3)

  def test_the_sentinel_fail_closes_into_a_recoverable_abort(self):
    action = generic_v2.parse_response(generic_v2.infer.ERROR_CALLING_LLM)

    self.assertEqual(action.action, generic_v2.ActionName.ABORT)
    self.assertTrue(generic_v2._is_recoverable_abort(action))


if __name__ == '__main__':
  absltest.main()
