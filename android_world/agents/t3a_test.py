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

from typing import Any
from unittest import mock

from absl.testing import absltest
from android_world.agents import infer
from android_world.agents import t3a
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
from android_world.utils import test_utils
import numpy as np


class MockLlmWrapper(infer.LlmWrapper):
  """Mock LLM wrapper for testing."""

  def __init__(self, mock_responses: list[tuple[str, Any]]):
    self.mock_responses = mock_responses
    self.index = 0

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Any]:
    if self.index < len(self.mock_responses):
      index = self.index
      self.index += 1
      return self.mock_responses[index][0], None, self.mock_responses[index][1]
    else:
      return infer.ERROR_CALLING_LLM, None, None


class T3AInteractionTest(absltest.TestCase):

  def test_step_method_with_completion(self):
    env = test_utils.FakeAsyncEnv()
    mock_llm = MockLlmWrapper([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        "fake_response",
    )])
    agent = t3a.T3A(env, mock_llm)

    goal = "do something"
    step_data = agent.step(goal)

    self.assertTrue(step_data.done)

  def test_history_recording(self):
    env = test_utils.FakeAsyncEnv()
    mock_llm = MockLlmWrapper([
        (
            (
                "Reason: completed.\nAction: {'action_type': 'answer',"
                " 'text': 'mock_response'}"
            ),
            "fake_response_1",
        ),
        (
            "fake_summary",
            "fake_response_1",
        ),
        (
            (
                "Reason: completed.\nAction: {'action_type': 'status',"
                " 'goal_status': 'complete'}"
            ),
            "fake_response_2",
        ),
    ])
    agent = t3a.T3A(env, mock_llm)

    goal = "do something"
    step1_data = agent.step(goal)
    self.assertFalse(step1_data.done)

    step2_data = agent.step(goal)
    self.assertTrue(step2_data.done)
    self.assertLen(agent.history, 2)


# The UI element list the fake server "returns" from /state. Built through the
# real serialization helpers so the fixtures match what a client actually
# decodes. Index 0 is a window frame with no text and index 1 is a clickable
# "Wi-Fi" row. is_visible must be set: the forest path on the server sets it,
# and m3a_utils.validate_ui_element drops elements where it is not True.
_UI_ELEMENTS = [
    representation_utils.ui_element_from_dict(
        representation_utils.ui_element_to_dict(
            representation_utils.UIElement(
                class_name='android.widget.FrameLayout',
                bbox=representation_utils.BoundingBox(0, 100, 0, 100),
                bbox_pixels=representation_utils.BoundingBox(0, 100, 0, 100),
                is_visible=True,
            )
        )
    ),
    representation_utils.ui_element_from_dict(
        representation_utils.ui_element_to_dict(
            representation_utils.UIElement(
                text='Wi-Fi',
                class_name='android.widget.TextView',
                bbox=representation_utils.BoundingBox(0, 50, 0, 20),
                bbox_pixels=representation_utils.BoundingBox(0, 50, 0, 20),
                is_clickable=True,
                is_visible=True,
            )
        )
    ),
]


class FakeAndroidEnvClient:
  """Duck-typed stand-in for interface.AndroidEnvClient.

  Implements only the surface ClientT3A uses, so tests never touch HTTP.
  """

  def __init__(self):
    self.reset_go_home = None
    self.hide_automation_ui_called = False
    self.executed_actions = []
    self.ui_elements = _UI_ELEMENTS

  def reset(self, go_home: bool) -> None:
    self.reset_go_home = go_home

  def hide_automation_ui(self) -> None:
    self.hide_automation_ui_called = True

  def get_state(self, wait_to_stabilize: bool = False) -> interface.State:
    del wait_to_stabilize
    return interface.State(
        pixels=np.zeros((100, 100, 3), dtype=np.uint8),
        forest=None,
        ui_elements=self.ui_elements,
        auxiliaries={},
    )

  def get_logical_screen_size(self) -> tuple[int, int]:
    return (100, 100)

  def get_physical_frame_boundary(self) -> tuple[int, int, int, int]:
    return (0, 0, 100, 100)

  def get_orientation(self) -> int:
    return 0

  def execute_action(self, action: json_action.JSONAction) -> None:
    self.executed_actions.append(action)


class ClientT3ATest(absltest.TestCase):

  def _create_agent(self, mock_responses: list[tuple[str, Any]]):
    client = FakeAndroidEnvClient()
    llm = MockLlmWrapper(mock_responses)
    return t3a.ClientT3A(client, llm), client, llm

  def test_step_method_with_completion(self):
    agent, _, _ = self._create_agent([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        "fake_response",
    )])

    step_data = agent.step("do something")

    self.assertTrue(step_data.done)
    self.assertLen(agent.history, 1)

  def test_completion_does_not_execute_an_action(self):
    agent, client, _ = self._create_agent([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        "fake_response",
    )])

    agent.step("do something")

    self.assertEmpty(client.executed_actions)

  def test_history_recording(self):
    agent, _, _ = self._create_agent([
        (
            (
                "Reason: completed.\nAction: {'action_type': 'answer',"
                " 'text': 'mock_response'}"
            ),
            "fake_response_1",
        ),
        (
            "fake_summary",
            "fake_response_1",
        ),
        (
            (
                "Reason: completed.\nAction: {'action_type': 'status',"
                " 'goal_status': 'complete'}"
            ),
            "fake_response_2",
        ),
    ])

    goal = "do something"
    step1_data = agent.step(goal)
    self.assertFalse(step1_data.done)

    step2_data = agent.step(goal)
    self.assertTrue(step2_data.done)
    self.assertLen(agent.history, 2)

  def test_step_executes_the_index_unchanged(self):
    # The server resolves the index against the same element list /state
    # returned, so the agent must forward it untouched.
    agent, client, _ = self._create_agent([
        (
            "Reason: open wifi.\nAction: {'action_type': 'click', 'index': 1}",
            "fake_response_1",
        ),
        ("fake_summary", "fake_response_1"),
    ])

    agent.step("open wifi")

    self.assertLen(client.executed_actions, 1)
    executed = client.executed_actions[0]
    self.assertEqual(executed.action_type, 'click')
    self.assertEqual(executed.index, 1)
    self.assertIsNone(executed.x)
    self.assertIsNone(executed.y)

  def test_step_rejects_out_of_range_index_without_executing(self):
    agent, client, _ = self._create_agent([(
        "Reason: bad index.\nAction: {'action_type': 'click', 'index': 42}",
        "fake_response_1",
    )])

    step_data = agent.step("do something")

    self.assertFalse(step_data.done)
    self.assertEmpty(client.executed_actions)
    self.assertIn('out of range', step_data.data['summary'])

  def test_step_survives_unparsable_action_output(self):
    agent, client, _ = self._create_agent([(
        "this is not the expected format",
        "fake_response_1",
    )])

    step_data = agent.step("do something")

    self.assertFalse(step_data.done)
    self.assertEmpty(client.executed_actions)
    self.assertLen(agent.history, 1)

  def test_step_survives_invalid_action_json(self):
    agent, client, _ = self._create_agent([(
        "Reason: nonsense.\nAction: {'action_type': 'teleport'}",
        "fake_response_1",
    )])

    step_data = agent.step("do something")

    self.assertFalse(step_data.done)
    self.assertEmpty(client.executed_actions)
    self.assertLen(agent.history, 1)

  def test_safety_classifier_marks_task_infeasible(self):
    client = FakeAndroidEnvClient()
    llm = mock.create_autospec(infer.LlmWrapper, instance=True)
    llm.predict.return_value = ('unsafe output', False, 'raw_response')
    agent = t3a.ClientT3A(client, llm)

    step_data = agent.step("do something")

    self.assertTrue(step_data.done)
    self.assertEmpty(client.executed_actions)

  def test_reset_hides_automation_ui_and_clears_history(self):
    agent, client, _ = self._create_agent([
        (
            "Reason: open wifi.\nAction: {'action_type': 'click', 'index': 1}",
            "fake_response_1",
        ),
        ("fake_summary", "fake_response_1"),
    ])
    agent.step("do something")
    self.assertLen(agent.history, 1)

    agent.reset(go_home_on_reset=True)

    self.assertTrue(client.hide_automation_ui_called)
    self.assertTrue(client.reset_go_home)
    self.assertEmpty(agent.history)

  def test_task_guidelines_are_included_in_the_prompt(self):
    agent, _, _ = self._create_agent([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        "fake_response",
    )])
    agent.set_task_guidelines(['Always use the open_app action.'])

    step_data = agent.step("do something")

    self.assertIn(
        'Always use the open_app action.', step_data.data['action_prompt']
    )

  def test_element_descriptions_come_from_the_server_state(self):
    agent, _, _ = self._create_agent([(
        (
            "Reason: completed.\nAction: {'action_type': 'status',"
            " 'goal_status': 'complete'}"
        ),
        "fake_response",
    )])

    step_data = agent.step("do something")

    # Assert on the element listing itself: "Wi-Fi" also appears in the
    # static guidance text, so a bare substring check would pass even if no
    # element were described.
    self.assertIn("UI element 1: UIElement(text='Wi-Fi'", step_data.data['action_prompt'])
    self.assertLen(step_data.data['before_element_list'], 2)


if __name__ == "__main__":
  absltest.main()
