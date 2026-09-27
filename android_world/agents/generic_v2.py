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

"""GenericAgentV2 pure-vision agent driving a Docker environment over HTTP.

This module is a self-contained reimplementation of the GenericAgentV2 core from
the `generic-v2-dynamicG-20260923-r2-sol` candidate repository (package
`generic_v2_harness`). It imports nothing from that repository: the system
prompt, the think/answer response parser, the bounded text history, the fact
ledger and the route-replan state machine are all inlined here.

The upstream trust boundary -- a JSONL worker process that cannot invoke a model
itself, with a separate adapter owning model credentials -- is deliberately NOT
reproduced. In this integration the agent *is* the adapter: it owns the model
call and drives `interface.AndroidEnvClient` directly.

Unlike the other Android World agents, the model here speaks normalized 0-1000
coordinates rather than UI element indices, so no accessibility element list is
needed. `_to_json_action` is the seam that turns those coordinates into the
pixel-space `json_action.JSONAction` the Docker server expects.
"""

import base64
import calendar
import dataclasses
import datetime
import enum
import hashlib
import json
import os
import re
import time

from android_world.agents import base_agent
from android_world.agents import infer
from android_world.env import interface
from android_world.env import json_action
import requests


# ---------------------------------------------------------------------------
# Action vocabulary
# ---------------------------------------------------------------------------


class ActionName(enum.StrEnum):
  """Action names the model is instructed to emit."""

  CLICK = 'CLICK'
  DOUBLE_TAP = 'DOUBLE_TAP'
  LONGPRESS = 'LONGPRESS'
  TYPE = 'TYPE'
  SWIPE = 'SWIPE'
  DRAG = 'DRAG'
  BACK = 'BACK'
  HOME = 'HOME'
  RECENT = 'RECENT'
  ENTER = 'ENTER'
  WAIT = 'WAIT'
  AWAKE = 'AWAKE'
  ANSWER = 'ANSWER'
  COMPLETE = 'COMPLETE'
  ABORT = 'ABORT'


@dataclasses.dataclass(frozen=True)
class ActionRequest:
  """One normalized model decision.

  Attributes:
    action: The requested action.
    payload: The full parsed JSON object from the model, unmodified apart from
      coordinate clamping. `_to_json_action` reads the action-specific fields
      out of it.
    thought: Contents of the model's `<THINK>` block.
    explain: The payload's `explain` field, if any.
    raw_response: The model response verbatim.
  """

  action: ActionName
  payload: dict
  thought: str
  explain: str
  raw_response: str


_TERMINAL_ACTIONS = frozenset(
    {ActionName.ABORT, ActionName.ANSWER, ActionName.COMPLETE}
)
# Terminal actions that translate to a `status` action, which the server treats
# as a no-op. Sending them is a wasted round trip, so they are skipped.
_STATUS_ACTIONS = frozenset({ActionName.ABORT, ActionName.COMPLETE})
# The baseline agent ends the episode only on COMPLETE and ABORT, mirroring the
# upstream `Action.is_terminal`. ANSWER is recorded by the environment and the
# agent keeps stepping -- unlike `ClientGenericR2SOL`, which treats it as
# terminal because an answer ends its evaluation.
_EPISODE_END_ACTIONS = frozenset({ActionName.ABORT, ActionName.COMPLETE})
_ROUTE_ACTIONS = frozenset(
    {ActionName.BACK, ActionName.HOME, ActionName.RECENT, ActionName.AWAKE}
)
_POINTER_ACTIONS = frozenset(
    {ActionName.CLICK, ActionName.DOUBLE_TAP, ActionName.LONGPRESS}
)

# ABORT payload values that mean "the response could not be parsed". These are
# recoverable: the episode should take another step rather than terminate.
_PARSE_ERROR_CODES = frozenset(
    {'empty_response', 'invalid_json', 'unknown_action'}
)
# ABORT payload value written when the route-replan escape already ran and the
# screen still did not change. This one is deliberately terminal.
_NO_PROGRESS_CODE = 'route_replan_no_progress'

# Longest client-side wait honored for a WAIT action, in seconds. The server's
# own `wait` action adds roughly one more second on top of this.
_MAX_WAIT_SECONDS = 10.0
_MAX_FACT_LEDGER_ENTRIES = 24
_MAX_FACT_KEY_CHARS = 100
_MAX_FACT_VALUE_CHARS = 500
_MAX_HISTORY_TURNS = 2


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """你是一个手机 GUI-Agent 操作专家。根据用户任务、当前手机截图和历史操作，
输出一个动作来完成任务。

坐标系左上角为原点，x 向右、y 向下，坐标范围均为 0-1000。

可用动作 JSON：
- CLICK / DOUBLE_TAP / LONGPRESS：{"action": "CLICK", "point": [x, y]}
- TYPE：{"action": "TYPE", "value": "文本", "point": [x, y], "clear": true}
- SWIPE / DRAG：{"action": "SWIPE", "point1": [x1, y1], "point2": [x2, y2]}
- BACK、HOME、RECENT、ENTER
- WAIT：{"action": "WAIT", "value": 秒数}
- AWAKE：{"action": "AWAKE", "value": "应用名称"}
- ANSWER：{"action": "ANSWER", "value": "答案"}
- COMPLETE：{"action": "COMPLETE", "return": "完成说明"}
- ABORT：{"action": "ABORT", "value": "原因"}

必须使用以下格式：
<THINK>简短描述当前界面、下一步和目标元素。</THINK>
<FACTS>{"字段名":"截图中与任务有关的原始字符串"}</FACTS>
<ANSWER>{"action": "动作类型"}</ANSWER>

FACTS 是跨步骤事实账本的增量：只有当前截图直接显示了之后需要回答、填写、发布或判断条件的事实
时才输出；逐字复制界面原文，不换算、不改写、不猜测。没有新事实时输出空对象 {}。后续消息中的
[事实账本]是已观察事实的唯一基准，不得用记忆或生成内容改写它。

要求：只依据任务、截图、已执行动作和事实账本决策。决策前先保留任务中关于对象类别、属性含义和
字段值类型或格式的全部明确限定，并将其视为硬约束而非偏好。对于向收件人发送消息的任务，先按任务
给出的目标语义选择通信方式：仅给出人名或账号的通用“发消息”请求，应优先使用已安装且以联系人/会话
为中心的聊天通信入口；只有任务明确指定短信，或收件人是电话号码时，才进入要求号码的短信入口。
某一通信入口查无此人或拒绝该标识，只能证明该入口不可用；截图或桌面显示还有其他符合目标语义的通信
入口时，必须至少尝试一个实质不同的入口，不能据此直接判定收件人不存在或使用 ABORT。任务含时间范围时，必须服从当前消息中的[时间目标]；该块给出的区间类型和边界是来源验收条件，
不得被界面中更方便的非等价摘要替代。规划导航时，区分任务要求的状态变更
和用于找路但会留下历史、记录或其他持久状态的附带变更。任务已限定目标所在的集合、关系或入口时，
必须先沿该限定路径查看；在该路径可用时，全局搜索或其他发现路径不算等效替代，不得先行使用。仅当
任务明确要求附带变更，或限定路径不可用且没有等效的非变更路径时，才使用会持久化状态的发现操作。
界面有多个候选或模式时，必须逐项对照全部限定；类别、字段语义或
格式有任一冲突就排除，不得因默认选中、位置靠前或表面相关而替代。对于要求开启、关闭或设为某状态的
任务，必须区分目标状态本身与“跟随”“自动”“继承”“由其他条件决定”等委托或条件模式：后者的开关
开启只证明该机制启用，不证明目标状态当前成立。若控件名称或说明表明结果取决于另一状态，当前页面即使
只显示该委托控件，也不得据此断定没有直接选项；必须先操作它退出委托模式，检查关闭、取消选择或切换
模式后显现的直接状态控件，并将其设为任务要求的状态。选中目标后若页面仍有“完成”“保存”“应用”“确认”
等提交控件，该选择只是待提交的界面状态；必须先操作提交控件，再依据提交后的截图确认目标状态仍成立，才能
COMPLETE。不得用提交前的勾选、相关能力已启用、默认选中、条件可能满足或“这是唯一可见选项”来推断完成。计数或判断标记状态
时，先逐项列出截图中实际可见的文字或图标标记；选中、高亮、当前位置等视觉状态不能替代缺失的目标
标记，也不得给未显示标记的项目补写状态。只把满足全部限定的候选作为目标事实或动作依据；若当前没有
符合项，继续检查其他选项或返回重选，不得猜测、填写或提交。坐标必须是 0-1000 的数字；JSON 必须有效；
完成所有任务后才用 COMPLETE。出现[动作效果核验]时，先比较动作前后截图中的目标区域，再决定下一步；出现
[无效动作警告]或[重复动作警告]时，不得把同一元素上的坐标微调当作新策略，必须重新聚焦、换用
不同操作方式或选择其他推进路径；出现[强制路径重规划]时，当前界面的本地交互路径已被执行层封锁，
只能离开当前界面并从不同屏幕或入口重规划，若无替代路径则 ABORT。"""


def _build_system_prompt() -> str:
  """Returns the system prompt stamped with the current date.

  The upstream implementation appends the date at import time, which goes stale
  in a long-running process; computing it per reset keeps `[当前日期]` truthful
  across episodes that span midnight.
  """
  return f'{_SYSTEM_PROMPT}\n\n[当前日期]\n{datetime.date.today().isoformat()}'


# ---------------------------------------------------------------------------
# Task-derived prompt gates
# ---------------------------------------------------------------------------

_ROLLING_PERIOD = re.compile(
    r'(?P<label>(?:最近|过去|近)(?P<count>\d+|[一二两三四五六七八九十两]+)?'
    r'(?P<unit>天|日|周|星期|个月|月|年))'
)
_CALENDAR_PERIOD = re.compile(r'(?P<label>本周|这周|本月|这个月|本年|今年)')
_CHINESE_DIGITS = {
    '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9,
}
_LOOKUP_TASK = re.compile(
    r'(?:查|查询|搜索|搜|查看|获取|find|look up|search)', re.IGNORECASE
)
_DURABLE_OUTPUT_TASK = re.compile(
    r'(?:保存|记录|写|填|发|提交|发送|创建|save|record|write|copy|post|submit|send|create)',
    re.IGNORECASE,
)
_SETTING_STATE = re.compile(
    r'(?:开启|打开|启用|允许|保留|保持|关闭|关掉|停用|禁用|隐藏|取消|设为|设置为|'
    r'enable|disable|turn\s+on|turn\s+off|keep|allow|block|hide|show|set\s+to)',
    re.IGNORECASE,
)


def _independent_settings_context(task: str) -> str:
  """Returns a current-turn completion gate for multi-state requests."""
  if len(_SETTING_STATE.findall(task)) < 2:
    return ''
  return (
      '[独立设置验收]\n'
      '任务包含多个目标状态。先按连词拆分所有对象，并在THINK开头保留简短的“对象→目标状态”'
      '清单；同一状态词修饰的并列对象也必须拆成独立项。先定位、后修改：在每一项的直接控件都已'
      '实际找到之前，只能导航、返回或滚动，任何设置开关都不得改动；即使先看到某一项，也先记住'
      '位置并继续寻找其余项。只有名称或说明明确表示同一对象的控件才算直接控件；过滤器、总开关、'
      '摘要、否定式隐藏项、近义类别或对其他开关状态的推断都不算。若某对象的直接状态被明确属于该对象'
      '的委托模式遮蔽，定位阶段只记录该委托入口及其路径，不得切换；所有对象的直接控件或这种专属入口'
      '都已定位后，才可退出委托模式以显现并设置直接状态。当前分支缺少任一项时，返回最近的分支页并'
      '检查尚未查看的兄弟入口，不得先改本页控件。全部路径定位后，只修改清单中状态不符的直接控件，'
      '不触碰清单外控件；每项均以修改后的截图或当前可见状态逐项核验后才可COMPLETE。'
  )


def _attribution_gate_context(task: str) -> str:
  """Returns a current-turn source-identity gate for lookup-and-copy tasks."""
  if not (_LOOKUP_TASK.search(task) and _DURABLE_OUTPUT_TASK.search(task)):
    return ''
  return (
      '[来源对象核验]\n'
      '只在准备把来源事实带到另一处写入、发布或提交的动作前执行一次核验：在THINK中用一句话逐字'
      '写出请求对象名和来源页显示名，并判断是否为同一对象范围；同时用字段名'
      '“请求=<任务原文对象>;显示=<来源页原文>;事实=<当前截图原文>”把已核验且之后要使用的原始事实'
      '写入FACTS。核验说明保持一句，不复述任务、规则或历史，然后照常输出ANSWER。\n'
      '尚在来源内搜索、选择、切换或打开详情时，不重复整段核验，也不把不匹配对象或推断说明写入'
      'FACTS；只简短说明下一导航动作。若显示名缺失，或是请求对象的成员、下级、上级、相关项或其他'
      '范围，必须留在来源内继续寻找，不能开始目标端写入。匹配名称虽已出现但所需属性尚未显示时，'
      '仍须打开其详情并读取属性后才能转移。进入目标端后可依据已经核验的事实账本完成写入，无需因'
      '目标界面不显示来源名而重复核验。仅字体、字形或文字体系差异且来源结果明确标识同一命名项目'
      '时，可视为同一对象。'
  )


def _temporal_goal_context(
    task: str, today: datetime.date | None = None
) -> str:
  """Returns an exact, action-oriented temporal source gate for a dated task."""
  current = today or datetime.date.today()
  rolling = _ROLLING_PERIOD.search(task)
  if rolling:
    count = _parse_count(rolling.group('count'))
    unit = rolling.group('unit')
    start = _rolling_start(current, count, unit)
    return _temporal_gate('滚动区间', rolling.group('label'), start, current)

  aligned = _CALENDAR_PERIOD.search(task)
  if aligned:
    label = aligned.group('label')
    if label in {'本周', '这周'}:
      start = current - datetime.timedelta(days=current.weekday())
      end = start + datetime.timedelta(days=6)
    elif label in {'本月', '这个月'}:
      start = current.replace(day=1)
      end = current.replace(
          day=calendar.monthrange(current.year, current.month)[1]
      )
    else:
      start = datetime.date(current.year, 1, 1)
      end = datetime.date(current.year, 12, 31)
    return _temporal_gate('日历对齐区间', label, start, end)
  return ''


def _parse_count(value: str | None) -> int:
  if not value:
    return 1
  if value.isdigit():
    return max(1, int(value))
  if value == '十':
    return 10
  if '十' in value:
    tens, ones = value.split('十', 1)
    return (10 if not tens else _CHINESE_DIGITS.get(tens, 1) * 10) + (
        0 if not ones else _CHINESE_DIGITS.get(ones, 0)
    )
  return _CHINESE_DIGITS.get(value, 1)


def _rolling_start(
    current: datetime.date, count: int, unit: str
) -> datetime.date:
  if unit in {'天', '日'}:
    return current - datetime.timedelta(days=count - 1)
  if unit in {'周', '星期'}:
    return current - datetime.timedelta(days=count * 7 - 1)
  if unit in {'个月', '月'}:
    return _shift_months(current, -count) + datetime.timedelta(days=1)
  try:
    return current.replace(year=current.year - count) + datetime.timedelta(
        days=1
    )
  except ValueError:  # February 29 shifted to a non-leap year.
    shifted = current.replace(year=current.year - count, day=28)
    return shifted + datetime.timedelta(days=1)


def _shift_months(value: datetime.date, offset: int) -> datetime.date:
  month_index = value.year * 12 + value.month - 1 + offset
  year, zero_based_month = divmod(month_index, 12)
  month = zero_based_month + 1
  day = min(value.day, calendar.monthrange(year, month)[1])
  return datetime.date(year, month, day)


def _temporal_gate(
    kind: str, label: str, start: datetime.date, end: datetime.date
) -> str:
  return (
      '[时间目标]\n'
      f'任务时间表达：{label}\n'
      f'区间类型：{kind}\n'
      f'必须覆盖（含首尾）：{start.isoformat()} 至 {end.isoformat()}\n'
      '来源验收门：统计、摘要或极值只有在截图可见日期范围与上述边界语义等价时才可写入'
      'FACTS并用于作答。范围缺失或不等价时，只能继续在来源中打开日历、历史、逐日明细或'
      '切换区间；不得离开来源去填写或提交，也不得因摘要已直接给出结论而放行。'
  )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_ACTION_ALIASES = {
    'TAP': 'CLICK',
    'DOUBLETAP': 'DOUBLE_TAP',
    'LONG_PRESS': 'LONGPRESS',
    'SLIDE': 'SWIPE',
    'LAUNCH': 'AWAKE',
    'FINISH': 'COMPLETE',
}

_COORDINATE_FIELDS = {
    ActionName.CLICK: ('point',),
    ActionName.DOUBLE_TAP: ('point',),
    ActionName.LONGPRESS: ('point',),
    ActionName.TYPE: ('point',),
    ActionName.SWIPE: ('point1', 'point2'),
    ActionName.DRAG: ('point1', 'point2'),
}

# Spellings the upstream action map accepts as synonyms for the canonical
# payload keys, as `(canonical, alias)` pairs. The baseline agent normalizes
# them after parsing; the R2-SOL core reads only the canonical keys.
_PAYLOAD_ALIASES = {
    ActionName.TYPE: (('value', 'text'),),
    ActionName.SWIPE: (('point1', 'start'), ('point2', 'end')),
    ActionName.DRAG: (('point1', 'start'), ('point2', 'end')),
    ActionName.WAIT: (('value', 'duration'),),
    ActionName.AWAKE: (('value', 'app'),),
    ActionName.ANSWER: (('value', 'text'),),
    ActionName.COMPLETE: (('return', 'message'),),
    ActionName.ABORT: (('value', 'reason'),),
}


def parse_response(response_text: str) -> ActionRequest:
  """Normalizes one model response into an action request.

  An invalid model answer maps to a fail-closed `ABORT` request whose payload
  `value` is a stable error code, so callers can tell an unparseable response
  from a deliberate abort.

  Args:
    response_text: The raw model response.

  Returns:
    The parsed action request.
  """
  raw = str(response_text or '').strip()
  if not raw:
    return _abort('empty_response', raw)
  thought = _extract_tag(raw, 'think')
  answer = _extract_tag(raw, 'answer') or raw
  parsed = _parse_first_json(answer)
  if not isinstance(parsed, dict):
    return _abort('invalid_json', raw, thought=thought)
  raw_action = str(
      parsed.get('action') or parsed.get('action_type') or ''
  ).strip().upper()
  try:
    action = ActionName(_ACTION_ALIASES.get(raw_action, raw_action))
  except ValueError:
    return _abort('unknown_action', raw, thought=thought)
  parsed = _clamp_action_coordinates(parsed, action)
  return ActionRequest(
      action=action,
      payload=parsed,
      thought=thought,
      explain=str(parsed.get('explain', '') or ''),
      raw_response=raw,
  )


def _extract_tag(value: str, name: str) -> str:
  match = re.search(
      rf'<{name}>(.*?)</{name}>', value, re.DOTALL | re.IGNORECASE
  )
  return match.group(1).strip() if match else ''


def _clamp_action_coordinates(
    payload: dict, action: ActionName
) -> dict:
  """Clamps pointer actions into the normalized 0-1000 screen bounds."""
  coordinate_fields = _COORDINATE_FIELDS.get(action, ())
  normalized = dict(payload)
  for field_name in coordinate_fields:
    point = normalized.get(field_name)
    if (
        isinstance(point, list)
        and len(point) == 2
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in point
        )
    ):
      normalized[field_name] = [min(1000, max(0, value)) for value in point]
  return normalized


def _parse_first_json(value: str):
  """Extracts the first balanced JSON object from text that may wrap it."""
  try:
    return json.loads(value)
  except json.JSONDecodeError:
    pass
  start = value.find('{')
  if start < 0:
    return None
  in_string = False
  escaped = False
  depth = 0
  for index, character in enumerate(value[start:], start=start):
    if in_string:
      if escaped:
        escaped = False
      elif character == '\\':
        escaped = True
      elif character == '"':
        in_string = False
    elif character == '"':
      in_string = True
    elif character == '{':
      depth += 1
    elif character == '}':
      depth -= 1
      if depth == 0:
        candidate = value[start : index + 1]
        try:
          return json.loads(candidate)
        except json.JSONDecodeError:
          repaired = _escape_bare_string_quotes(candidate)
          try:
            return json.loads(repaired)
          except json.JSONDecodeError:
            return None
  return None


def _escape_bare_string_quotes(value: str) -> str:
  """Escapes quotes that cannot terminate their current JSON string.

  This narrowly recovers otherwise valid action objects when prose inside a
  string contains literal quotation marks. Structural quotes (before a colon,
  comma, closing brace, or closing bracket) are left unchanged.

  Args:
    value: A candidate JSON object string.

  Returns:
    The string with non-structural interior quotes escaped.
  """
  output = []
  in_string = False
  escaped = False
  for index, character in enumerate(value):
    if escaped:
      output.append(character)
      escaped = False
      continue
    if character == '\\' and in_string:
      output.append(character)
      escaped = True
      continue
    if character != '"':
      output.append(character)
      continue
    if not in_string:
      in_string = True
      output.append(character)
      continue
    following = value[index + 1 :].lstrip()
    if not following or following[0] in ':,}]':
      in_string = False
      output.append(character)
    else:
      output.append('\\"')
  return ''.join(output)


def _abort(error: str, raw: str, *, thought: str = '') -> ActionRequest:
  return ActionRequest(
      action=ActionName.ABORT,
      payload={'action': 'ABORT', 'value': error},
      thought=thought,
      explain='',
      raw_response=raw,
  )


# ---------------------------------------------------------------------------
# Agent core: message construction, bounded memory, route replanning
# ---------------------------------------------------------------------------


class GenericV2Core:
  """Stateful pure-vision decision core, independent of the environment.

  Holds no model client and performs no I/O. `prepare`-style message building
  and `interpret` are a stateful pair: `build_messages` records the current
  screenshot so that the *next* turn can show before/after images side by side.
  """

  def __init__(self, model_args: dict | None = None):
    """Initializes the core.

    Args:
      model_args: Sampling parameters passed to the model. Defaults to the
        upstream values.
    """
    self.model_args = dict(
        model_args
        or {
            'temperature': 0.1,
            'top_p': 0.95,
            'frequency_penalty': 0.0,
            'max_tokens': 32768,
        }
    )
    self.task = ''
    self.temporal_context = ''
    self.history = []
    self.fact_ledger = {}
    self.previous_image_data_url = ''
    self.pending_image_data_url = ''
    self.unchanged_state_streak = 0
    self.feedback_step_index = -1
    self.trajectory_step_index = -1
    self.recent_state_signatures = []
    self.recent_action_signatures = []
    self.attribution_gate_active = False
    self.route_replan_required = False
    self.route_escape_attempted = False

  def reset(self, task: str) -> None:
    """Starts a new task, clearing all cross-step memory."""
    self.task = task
    self.temporal_context = _temporal_goal_context(task)
    self.history = []
    self.fact_ledger = {}
    self.previous_image_data_url = ''
    self.pending_image_data_url = ''
    self.unchanged_state_streak = 0
    self.feedback_step_index = -1
    self.trajectory_step_index = -1
    self.recent_state_signatures = []
    self.recent_action_signatures = []
    self.attribution_gate_active = bool(_attribution_gate_context(task))
    self.route_replan_required = False
    self.route_escape_attempted = False

  def build_messages(
      self, step_index: int, image_data_url: str
  ) -> list[dict]:
    """Builds the chat messages for one decision.

    Mutates progress-tracking state (recent state signatures and the
    no-progress streak) but is safe to call twice for the same `step_index`.

    Args:
      step_index: Zero-based index of this step within the episode.
      image_data_url: The current screenshot as a data URL.

    Returns:
      Messages in OpenAI chat format.

    Raises:
      RuntimeError: If `reset` has not been called.
    """
    if not self.task:
      raise RuntimeError('reset(task) must be called before build_messages()')
    messages = [
        {'role': 'system', 'content': _build_system_prompt()}
    ]
    for index, response in enumerate(self.history):
      text = f'[任务]\n{self.task}' if index == 0 else f'[Step {index + 1}]'
      messages.extend(
          (
              {'role': 'user', 'content': [{'type': 'text', 'text': text}]},
              {'role': 'assistant', 'content': response},
          )
      )
    step = step_index + 1
    user_text = f'[任务]\n{self.task}' if not self.history else f'[Step {step}]'
    if self.temporal_context:
      user_text += f'\n{self.temporal_context}'
    if self.fact_ledger:
      ledger = json.dumps(
          self.fact_ledger, ensure_ascii=False, separators=(',', ':')
      )
      user_text += f'\n[事实账本]\n{ledger}'
    if self.attribution_gate_active:
      user_text += f'\n{_attribution_gate_context(self.task)}'
    settings_context = _independent_settings_context(self.task)
    if settings_context:
      user_text += f'\n{settings_context}'
    current_content = [
        {'type': 'image_url', 'image_url': {'url': image_data_url}},
        {'type': 'text', 'text': user_text},
    ]
    if step_index != self.trajectory_step_index:
      self.recent_state_signatures.append(
          hashlib.sha256(image_data_url.encode()).hexdigest()
      )
      self.recent_state_signatures[:] = self.recent_state_signatures[-7:]
      self.trajectory_step_index = step_index
    if self.previous_image_data_url:
      state_unchanged = self.previous_image_data_url == image_data_url
      if step_index != self.feedback_step_index:
        self.unchanged_state_streak = (
            self.unchanged_state_streak + 1 if state_unchanged else 0
        )
        if not state_unchanged:
          self.route_replan_required = False
          self.route_escape_attempted = False
        elif self.unchanged_state_streak >= 2:
          if not self.route_replan_required:
            self.route_escape_attempted = False
          self.route_replan_required = True
        self.feedback_step_index = step_index
      last_action = parse_response(self.history[-1])
      feedback = (
          '[动作效果核验]\n'
          f'待核验的上一动作：'
          f'{json.dumps(last_action.payload, ensure_ascii=False)}'
      )
      repeated_action = len(self.history) > 1 and _action_signature(
          self.history[-2]
      ) == _action_signature(self.history[-1])
      if state_unchanged:
        feedback += (
            '\n[无效动作警告] 动作前后截图完全相同，上一动作没有产生可见效果。'
            '不得仅通过微调坐标继续尝试同一界面元素；下一动作必须重新聚焦、改用不同'
            '操作方式，或选择其他可推进任务的路径。'
        )
      if repeated_action:
        feedback += (
            '\n[重复动作警告] 最近两个动作和参数完全相同。除非当前状态明确显示持续'
            '进展，否则不得再次重复。'
        )
      if self.route_replan_required:
        feedback += (
            f'\n[强制路径重规划] 已连续 {self.unchanged_state_streak} 个动作未改变界面。'
            '当前界面的坐标交互路径已临时封锁，直到截图出现可见进展前，不得输出 CLICK、'
            'DOUBLE_TAP、LONGPRESS、TYPE、SWIPE 或 DRAG，也不得尝试同一控件的其他坐标。'
            '必须用 BACK、HOME、RECENT 或 AWAKE 离开当前界面，从不同屏幕或入口重新规划；'
            '若不存在合理替代路径则 ABORT。其他非终止动作将由执行层改为 BACK；若该次逃逸'
            '仍未改变界面，执行层将终止而不再重试。'
        )
      cycle_period = _repeated_cycle_period(
          self.recent_state_signatures, self.recent_action_signatures
      )
      if cycle_period:
        feedback += (
            '\n[轨迹循环警告] 最近的状态/动作轨迹或手势模式形成了重复的 '
            f'{cycle_period} 步循环。'
            '立即停止当前动作模式；重新判断控件几何、当前状态和可用路径，并在继续重试前'
            '选择操作方式、目标区域或推进路径上实质不同的策略。'
        )
      current_content = [
          {'type': 'text', 'text': '[执行上一动作前的状态]'},
          {
              'type': 'image_url',
              'image_url': {'url': self.previous_image_data_url},
          },
          {'type': 'text', 'text': '[执行上一动作后的当前状态]'},
          *current_content,
          {'type': 'text', 'text': feedback},
      ]
    messages.append({'role': 'user', 'content': current_content})
    return messages

  def interpret(self, response: str) -> ActionRequest:
    """Parses one model response and retains bounded memory of it.

    May rewrite the requested action to BACK or ABORT when the route-replan
    state machine has engaged. The rewritten decision is also written back into
    the retained history, so the model's own transcript stays self-consistent
    with what was actually executed.

    Args:
      response: The raw model response.

    Returns:
      The action to execute.
    """
    action = parse_response(response)
    self._remember_facts(response)
    history_response = response
    if _retries_same_pointer_target(self.recent_action_signatures, action):
      self.route_replan_required = True
      self.route_escape_attempted = False
    if self.route_replan_required and action.action not in _TERMINAL_ACTIONS:
      if self.route_escape_attempted:
        action = ActionRequest(
            action=ActionName.ABORT,
            payload={'action': 'ABORT', 'value': _NO_PROGRESS_CODE},
            thought=(
                f'{action.thought}\n[执行约束] 强制离开当前路径后界面仍无可见进展，'
                '停止继续消耗动作预算。'
            ).strip(),
            explain='',
            raw_response=response,
        )
      elif action.action not in _ROUTE_ACTIONS:
        action = ActionRequest(
            action=ActionName.BACK,
            payload={'action': 'BACK'},
            thought=(
                f'{action.thought}\n[执行约束] 当前无进展界面的本地交互路径已封锁，'
                '改为 BACK 以从不同屏幕重新规划。'
            ).strip(),
            explain='',
            raw_response=response,
        )
        self.route_escape_attempted = True
      else:
        self.route_escape_attempted = True
      history_response = (
          f'<THINK>{action.thought}</THINK>\n<FACTS>{{}}</FACTS>\n'
          f'<ANSWER>{json.dumps(action.payload, ensure_ascii=False)}</ANSWER>'
      )
    self.recent_action_signatures.append(
        json.dumps(
            action.payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        )
    )
    self.recent_action_signatures[:] = self.recent_action_signatures[-6:]
    if self.pending_image_data_url:
      self.previous_image_data_url = self.pending_image_data_url
      self.pending_image_data_url = ''
    self.history.append(history_response)
    # Keep one screenshot for action-effect comparison while bounding text
    # history to the two most recent model turns.
    if len(self.history) > _MAX_HISTORY_TURNS:
      self.history[:] = self.history[-_MAX_HISTORY_TURNS:]
    return action

  def remember_observation_image(self, image_data_url: str) -> None:
    """Records the screenshot the current decision was based on."""
    self.pending_image_data_url = image_data_url

  def _remember_facts(self, response: str) -> None:
    """Retains model-declared observations outside the rolling history."""
    match = re.search(
        r'<facts>(.*?)</facts>', response, re.DOTALL | re.IGNORECASE
    )
    if not match:
      return
    try:
      facts = json.loads(match.group(1))
    except (json.JSONDecodeError, TypeError):
      return
    if not isinstance(facts, dict):
      return
    for field_name, observed_value in facts.items():
      if (
          len(self.fact_ledger) >= _MAX_FACT_LEDGER_ENTRIES
          and field_name not in self.fact_ledger
      ):
        break
      if (
          isinstance(field_name, str)
          and isinstance(observed_value, str)
          and field_name
          and observed_value
          and len(field_name) <= _MAX_FACT_KEY_CHARS
          and len(observed_value) <= _MAX_FACT_VALUE_CHARS
      ):
        self.fact_ledger[field_name] = observed_value


def _action_signature(response: str) -> str:
  """Returns a stable signature for repetition feedback in the next prompt."""
  action = parse_response(response)
  return json.dumps(
      action.payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')
  )


def _retries_same_pointer_target(
    recent_actions: list[str], current_action: ActionRequest
) -> bool:
  """Detects a third consecutive tap retry within one small target region."""
  if (
      current_action.action not in _POINTER_ACTIONS
      or len(recent_actions) < 2
  ):
    return False
  payloads = []
  for encoded in [
      *recent_actions[-2:],
      json.dumps(current_action.payload),
  ]:
    try:
      payload = json.loads(encoded)
    except (json.JSONDecodeError, TypeError):
      return False
    if str(payload.get('action', '')).upper() not in {
        'CLICK',
        'DOUBLE_TAP',
        'LONGPRESS',
    }:
      return False
    point = payload.get('point')
    if (
        not isinstance(point, list)
        or len(point) != 2
        or not all(isinstance(value, (int, float)) for value in point)
    ):
      return False
    payloads.append(payload)
  points = [payload['point'] for payload in payloads]
  return all(
      (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2 <= 80**2
      for index, left in enumerate(points)
      for right in points[index + 1 :]
  )


def _repeated_cycle_period(states: list[str], actions: list[str]) -> int:
  """Returns a short repeated trajectory period, or zero when non-cyclic."""
  for period in range(1, 4):
    transition_count = period * 2
    if len(states) < transition_count + 1 or len(actions) < transition_count:
      continue
    recent_states = states[-(transition_count + 1) :]
    recent_actions = actions[-transition_count:]
    actions_repeat = recent_actions[:period] == recent_actions[period:]
    states_repeat = (
        recent_states[:period] == recent_states[period:transition_count]
        and recent_states[0] == recent_states[-1]
    )
    alternating_actions = (
        period == 2 and len(set(recent_actions[:period])) == period
    )
    if actions_repeat and (states_repeat or alternating_actions):
      return period
  return 0


# ---------------------------------------------------------------------------
# Action translation
# ---------------------------------------------------------------------------


def _normalized_to_pixels(
    point, logical_screen_size: tuple[int, int]
) -> tuple[int, int] | None:
  """Converts one 0-1000 normalized point to device pixels."""
  if (
      not isinstance(point, list)
      or len(point) != 2
      or not all(isinstance(value, (int, float)) for value in point)
  ):
    return None
  width, height = logical_screen_size
  x = int(min(1000.0, max(0.0, float(point[0]))) / 1000 * width)
  y = int(min(1000.0, max(0.0, float(point[1]))) / 1000 * height)
  return x, y


_DIRECTIONS = ('left', 'right', 'up', 'down')


def _swipe_direction(
    start: tuple[int, int] | None, end: tuple[int, int] | None
) -> str | None:
  """Returns the finger-travel direction from start to end.

  Args:
    start: Pixel touch point, or None.
    end: Pixel lift point, or None.

  Returns:
    One of 'left', 'right', 'up', 'down', or None when the span is degenerate.
  """
  if start is None or end is None:
    return None
  dx = end[0] - start[0]
  dy = end[1] - start[1]
  if max(abs(dx), abs(dy)) < 1:
    return None
  if abs(dy) >= abs(dx):
    return 'down' if dy > 0 else 'up'
  return 'right' if dx > 0 else 'left'


def _payload_value(payload: dict) -> str:
  """Returns an action's scalar argument as text, or an empty string."""
  value = payload.get('value')
  if value is None:
    value = payload.get('return')
  if isinstance(value, bool) or value is None:
    return ''
  return str(value)


def _wait_seconds(value) -> float:
  """Returns the bounded client-side delay for a WAIT action.

  Mirrors the upstream extractor `float(value or duration or 1.0)`: a missing
  value means one second. A malformed value raises, and the caller turns that
  into a failed step rather than a wrong sleep.

  Args:
    value: The requested duration, possibly absent.

  Returns:
    Seconds to sleep, clamped to `_MAX_WAIT_SECONDS`.
  """
  if value is None:
    return 1.0
  return min(max(float(str(value).strip()), 0.0), _MAX_WAIT_SECONDS)


def _to_json_action(
    action: ActionRequest, logical_screen_size: tuple[int, int]
) -> json_action.JSONAction | None:
  """Translates one normalized action into the client's action vocabulary.

  Returns None for `RECENT`, which has no `json_action` equivalent and is
  handled separately through `press_key`.

  Args:
    action: The action request to translate.
    logical_screen_size: Device logical screen size in pixels.

  Returns:
    The translated action, or None when the caller must special-case it.

  Raises:
    ValueError: When the action's payload lacks the fields it needs.
  """
  name = action.action
  payload = action.payload

  if name == ActionName.RECENT:
    return None

  if name in _POINTER_ACTIONS:
    point = _normalized_to_pixels(payload.get('point'), logical_screen_size)
    if point is None:
      raise ValueError(f'{name} requires a valid point')
    action_type = {
        ActionName.CLICK: json_action.CLICK,
        ActionName.DOUBLE_TAP: json_action.DOUBLE_TAP,
        ActionName.LONGPRESS: json_action.LONG_PRESS,
    }[name]
    return json_action.JSONAction(
        action_type=action_type, x=point[0], y=point[1]
    )

  if name == ActionName.TYPE:
    point = _normalized_to_pixels(payload.get('point'), logical_screen_size)
    if point is None:
      raise ValueError('TYPE requires a valid point')
    if not _payload_value(payload):
      raise ValueError('TYPE requires non-empty text')
    # Returns only the focus click; `_execute` does the typing. The server's
    # `input_text` always presses ENTER after typing, which would submit the
    # field prematurely -- the model emits ENTER explicitly when it wants that.
    return json_action.JSONAction(
        action_type=json_action.CLICK, x=point[0], y=point[1]
    )

  if name in (ActionName.SWIPE, ActionName.DRAG):
    start = _normalized_to_pixels(payload.get('point1'), logical_screen_size)
    end = _normalized_to_pixels(payload.get('point2'), logical_screen_size)
    declared = str(payload.get('direction') or '').strip().lower()
    if name == ActionName.SWIPE:
      # A swipe is a finger-travel gesture (scroll/fling), which the server's
      # `swipe` serves as a 500ms full-screen fling. `drag_and_drop` is
      # `input draganddrop` with a 4000ms hold -- a long press, which on a list
      # item opens a context menu rather than scrolling the list.
      direction = declared if declared in _DIRECTIONS else _swipe_direction(
          start, end
      )
      if direction:
        return json_action.JSONAction(
            action_type=json_action.SWIPE, direction=direction
        )
    if start is None or end is None:
      raise ValueError(f'{name} requires point1 and point2')
    return json_action.JSONAction(
        action_type=json_action.DRAG_AND_DROP,
        touch_xy=[start[0], start[1]],
        lift_xy=[end[0], end[1]],
    )

  if name == ActionName.BACK:
    return json_action.JSONAction(action_type=json_action.NAVIGATE_BACK)
  if name == ActionName.HOME:
    return json_action.JSONAction(action_type=json_action.NAVIGATE_HOME)
  if name == ActionName.ENTER:
    return json_action.JSONAction(action_type=json_action.KEYBOARD_ENTER)
  if name == ActionName.WAIT:
    return json_action.JSONAction(action_type=json_action.WAIT)

  if name == ActionName.AWAKE:
    app_name = _payload_value(payload)
    if not app_name:
      raise ValueError('AWAKE requires an app name')
    return json_action.JSONAction(
        action_type=json_action.OPEN_APP, app_name=app_name
    )

  if name == ActionName.ANSWER:
    return json_action.JSONAction(
        action_type=json_action.ANSWER, text=_payload_value(payload)
    )
  if name == ActionName.COMPLETE:
    return json_action.JSONAction(
        action_type=json_action.STATUS, goal_status='complete'
    )
  if name == ActionName.ABORT:
    return json_action.JSONAction(
        action_type=json_action.STATUS, goal_status='infeasible'
    )

  raise ValueError(f'Unsupported action: {name}')


def _is_recoverable_abort(action: ActionRequest) -> bool:
  """Whether an ABORT merely means "the model response was unusable"."""
  if action.action != ActionName.ABORT:
    return False
  return str(action.payload.get('value', '')) in _PARSE_ERROR_CODES


def _message_text(content) -> str:
  """Returns a chat completion's text, treating a null content as empty.

  Thinking models served by vLLM report a null `content` when the turn produced
  only reasoning, and a JSON body can carry the same null. An empty string then
  fail-closes into a recoverable ABORT instead of crashing the episode.

  Args:
    content: The completion's `content` field.

  Returns:
    The text, or an empty string.
  """
  return content if isinstance(content, str) else ''


def _render_text_only(messages: list[dict]) -> str:
  """Flattens chat messages to text, replacing images with a placeholder.

  Episode logs serialise `step_data`, so base64 screenshots must never be
  stored there.
  """
  parts = []
  for message in messages:
    role = message.get('role', '')
    content = message.get('content')
    if isinstance(content, str):
      parts.append(f'[{role}] {content}')
      continue
    for item in content or []:
      if item.get('type') == 'text':
        parts.append(f'[{role}] {item.get("text", "")}')
      else:
        parts.append(f'[{role}] <image>')
  return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ClientGeneric(base_agent.ClientInteractingAgent):
  """GenericAgentV2 pure-vision agent driving a Docker environment over HTTP.

  Speaks normalized 0-1000 coordinates rather than UI element indices, so it
  needs no accessibility element list and never resolves an index. The model is
  called directly so the upstream sampling parameters survive: this agent asks
  for `max_tokens: 32768`, which `infer.OpenAIWrapper` hardcodes to 1000.

  Transport follows `infer.OpenAIWrapper`: the `openai` SDK when importable,
  raw `requests` otherwise, with the same exponential backoff.

  This class is a faithful port of the upstream *baseline* `GenericAgentV2`
  (MobileGym's `bench_env/agent/generic_v2.py`): the same Chinese system prompt,
  the same think/answer parsing, the same action aliases and payload spellings,
  and the same replay of the full text history. It deliberately has none of the
  R2-SOL machinery -- no `<FACTS>` ledger, no temporal/source/settings gates, no
  route replanning -- so it can serve as the baseline that `ClientGenericR2SOL`
  is measured against.
  """

  SYSTEM_PROMPT = """你是一个手机 GUI-Agent 操作专家。你需要根据用户下发的任务、手机屏幕截图以及历史操作记录，分析当前界面并输出一个动作来与手机交互，从而完成任务。

坐标系：左上角为原点，x 向右，y 向下，取值范围均为 0-1000（归一化坐标）。

可用动作（JSON 格式）：

1. 点击：{"action": "CLICK", "point": [x, y]}
2. 双击：{"action": "DOUBLE_TAP", "point": [x, y]}
3. 长按：{"action": "LONGPRESS", "point": [x, y]}
4. 输入：{"action": "TYPE", "value": "文本内容"}  // 可选 "point": [x, y] 指定输入位置；可选 "clear": true 先清空输入框再输入（默认追加到已有文本后面）
5. 滑动：{"action": "SWIPE", "point1": [x1, y1], "point2": [x2, y2]}
6. 拖拽：{"action": "DRAG", "point1": [x1, y1], "point2": [x2, y2]}  // 按住起点拖动到终点
7. 返回：{"action": "BACK"}
8. 回到桌面：{"action": "HOME"}
9. 打开最近任务：{"action": "RECENT"}
10. 输入回车：{"action": "ENTER"}
11. 等待：{"action": "WAIT", "value": 秒数}
12. 打开应用：{"action": "AWAKE", "value": "应用名称"}
13. 提交答案：{"action": "ANSWER", "value": "纯答案文本"}
14. 任务完成：{"action": "COMPLETE", "return": "完成说明"} // 所有任务完成后使用，给出简短的说明
15. 中止任务：{"action": "ABORT", "value": "中止原因"}  // 任务无法完成时使用，需要说明原因


你必须按以下格式输出：

<THINK>
在这里描述你对当前屏幕的理解、分析和决策过程。
包括：
1. 当前屏幕显示的内容是什么
2. 为了完成任务，下一步应该做什么
3. 具体要点击/操作哪个元素
</THINK>
<ANSWER>
{
  "action": "动作类型",
  // 根据动作类型填写相应参数
}
</ANSWER>


要求：
- 坐标必须为数字，范围 0-1000
- JSON 必须是有效格式
- 仔细观察屏幕截图，根据视觉信息做出判断
- 需要回答问题时，必须使用 ANSWER 提交答案
- COMPLETE 只用于结束任务，需要在执行完任务后使用
"""

  DEFAULT_MODEL_ARGS = {
      'temperature': 0.1,
      'top_p': 0.95,
      'frequency_penalty': 0.0,
      'max_tokens': 32768,
  }

  RETRY_WAITING_SECONDS = 20
  MAX_RETRY_CEILING = 5
  # Longer than OpenAIWrapper's 60s: a decision turn carries a 1080x2400 JPEG
  # data URL and may emit up to 32768 tokens.
  MODEL_TIMEOUT_SEC = 180

  def __init__(
      self,
      client: interface.AndroidEnvClient,
      base_url: str,
      model_name: str,
      max_retry: int = 3,
      name: str = 'ClientGeneric',
      model_args: dict | None = None,
  ):
    """Initializes the agent.

    Args:
      client: The Android environment client.
      base_url: Base URL of the OpenAI-compatible model endpoint. A trailing
        `/v1` is tolerated on both transports.
      model_name: Model name to request.
      max_retry: Max number of retries when a model call fails.
      name: The agent name.
      model_args: Sampling parameters, merged over `DEFAULT_MODEL_ARGS` the way
        the upstream agent merges its `config.model_args`.

    Raises:
      RuntimeError: If OPENAI_API_KEY is not set in the environment.
    """
    super().__init__(client, name)
    if 'OPENAI_API_KEY' not in os.environ:
      raise RuntimeError('OpenAI API key not set.')
    self._api_key = os.environ['OPENAI_API_KEY']
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, self.MAX_RETRY_CEILING)
    self._base_url = base_url
    self._model_name = model_name
    self.client_sdk = None
    try:
      from openai import OpenAI
      self.client_sdk = OpenAI(api_key=self._api_key, base_url=self._base_url)
    except ImportError:
      print(
          'OpenAI package not installed. Falling back to requests for model'
          ' calls.'
      )
    merged_args = dict(self.DEFAULT_MODEL_ARGS)
    if model_args:
      merged_args.update(model_args)
    self.model_args = merged_args
    self._task = ''
    # Raw model responses, oldest first. Unlike the R2-SOL core's bounded
    # history, the upstream agent replays every past turn -- its own memory
    # slimming only drops observations and prompts, never the text.
    self._responses = []
    self.history = []

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    self.client.hide_automation_ui()
    self.history = []
    self._responses = []
    self._task = ''

  def get_post_transition_state(self) -> interface.State:
    """Gets the state, waiting for the screen to settle after an action.

    Returns:
      The state with pixels and the server-side UI elements. The accessibility
      forest stays in the container, so it is None.
    """
    if self.transition_pause is None:
      print('Waiting for screen to stabilize before grabbing state...')
      start = time.time()
      state = self.client.get_state(wait_to_stabilize=True)
      print('Fetched after %.1f seconds.', time.time() - start)
      return state
    else:
      time.sleep(self.transition_pause)
      return self.client.get_state(wait_to_stabilize=False)

  def _chat_url(self) -> str:
    """Returns the chat-completions URL, tolerating a trailing `/v1`.

    `run_on_docker.py --base_url` is conventionally given with a trailing `/v1`
    (see scripts/run_on_docker.sh), and the `openai` SDK wants it that way, but
    appending the path unconditionally would yield `/v1/v1/chat/completions`.
    """
    base = self._base_url.rstrip('/')
    if base.endswith('/v1'):
      return f'{base}/chat/completions'
    return f'{base}/v1/chat/completions'

  def _call_model(self, messages: list[dict], model_args: dict) -> str:
    """Calls the OpenAI-compatible endpoint, retrying transient failures.

    Mirrors `infer.OpenAIWrapper`: the `openai` SDK when available, a raw
    `requests` POST otherwise, both retried with exponential backoff. Unlike
    the wrapper, the caller's own `messages` and sampling parameters are passed
    through unchanged.

    Args:
      messages: Chat messages, possibly carrying image data URLs.
      model_args: Sampling parameters.

    Returns:
      The model's text response, or `infer.ERROR_CALLING_LLM` on total failure.
      That sentinel is unparseable, so it fail-closes into a recoverable ABORT
      and the episode takes another step. A plain string is returned rather
      than OpenAIWrapper's `(text, is_safe, raw)` triple because `step_data` is
      pickled into checkpoints and must stay serializable.
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self._api_key}',
    }
    payload = {'model': self._model_name, 'messages': messages, **model_args}
    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        if self.client_sdk is None:
          response = requests.post(
              self._chat_url(),
              headers=headers,
              json=payload,
              timeout=self.MODEL_TIMEOUT_SEC,
          )
          if response.ok and 'choices' in response.json():
            return _message_text(
                response.json()['choices'][0]['message']['content']
            )
          print(
              f'Error calling model (HTTP {response.status_code}):'
              f' {response.text[:500]}'
          )
        else:
          response = self.client_sdk.chat.completions.create(
              model=self._model_name,
              messages=messages,
              timeout=self.MODEL_TIMEOUT_SEC,
              **model_args,
          )
          if response and response.choices:
            return _message_text(response.choices[0].message.content)
          print(f'Error calling model with error message: {response}')
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        print('Error calling LLM, will retry soon...')
        print(e)
      # Decremented on every failed attempt, including HTTP-level errors;
      # OpenAIWrapper only decrements on exceptions, which can spin forever on
      # a consistently failing endpoint.
      counter -= 1
      if counter > 0:
        time.sleep(wait_seconds)
        wait_seconds *= 2
    return infer.ERROR_CALLING_LLM

  def _build_messages(self, image_data_url: str) -> list[dict]:
    """Builds the chat messages for one decision.

    Mirrors the upstream `build_messages`: the task opens the first user turn,
    every past turn is replayed as text (the task, then `[Step N]`) followed by
    the model's raw response, and only the current turn carries the screenshot.

    Args:
      image_data_url: The current screenshot as a data URL.

    Returns:
      Messages in OpenAI chat format.
    """
    messages = [
        {'role': 'system', 'content': self.SYSTEM_PROMPT}
    ]
    for index, response in enumerate(self._responses):
      text = f'[任务]\n{self._task}' if index == 0 else f'[Step {index + 1}]'
      messages.append(
          {'role': 'user', 'content': [{'type': 'text', 'text': text}]}
      )
      messages.append({'role': 'assistant', 'content': response})
    step = len(self._responses) + 1
    user_text = (
        f'[任务]\n{self._task}' if not self._responses else f'[Step {step}]'
    )
    messages.append({
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': image_data_url}},
            {'type': 'text', 'text': user_text},
        ],
    })
    return messages

  def _normalize_payload(self, action: ActionRequest) -> ActionRequest:
    """Applies the upstream action map's payload synonyms to one request.

    The upstream agent folds these into its action-name -> payload extractors;
    running them right after parsing keeps the same model-facing vocabulary
    without touching the shared translation seam.

    Args:
      action: The parsed action request.

    Returns:
      The request with canonical keys filled in where only a synonym was given.
    """
    aliases = _PAYLOAD_ALIASES.get(action.action)
    if not aliases:
      return action
    payload = dict(action.payload)
    for canonical, alias in aliases:
      if payload.get(canonical) is None and payload.get(alias) is not None:
        payload[canonical] = payload[alias]
    return dataclasses.replace(action, payload=payload)

  def _execute(
      self, action: ActionRequest, screen_size: tuple[int, int]
  ) -> None:
    """Executes one action against the environment.

    Args:
      action: The normalized action.
      screen_size: Device logical screen size in pixels.

    Raises:
      ValueError: When the action's payload cannot be translated.
      Exception: Whatever the client raises when execution fails server-side.
    """
    if action.action == ActionName.RECENT:
      # No `json_action` equivalent exists; the recents key is the closest.
      self.client.press_key('KEYCODE_APP_SWITCH')
      return
    if action.action == ActionName.WAIT:
      # JSONAction has no duration field, so honor the requested delay here.
      # The server's own `wait` adds roughly one more second on top.
      delay = _wait_seconds(action.payload.get('value'))
      if delay > 0:
        time.sleep(delay)
      self.client.execute_action(
          json_action.JSONAction(action_type=json_action.WAIT)
      )
      return
    if action.action == ActionName.TYPE:
      # The upstream prompt makes `point` optional, so the field is only focused
      # when one was given. Typing deliberately avoids the server's
      # `input_text`, which always presses ENTER and would submit the field
      # prematurely -- the model emits ENTER explicitly when it wants that.
      value = _payload_value(action.payload)
      if not value:
        raise ValueError('TYPE requires non-empty text')
      point = _normalized_to_pixels(action.payload.get('point'), screen_size)
      if point is not None:
        self.client.execute_action(
            json_action.JSONAction(
                action_type=json_action.CLICK, x=point[0], y=point[1]
            )
        )
        time.sleep(1.0)
      if action.payload.get('clear'):
        # `clear` mirrors what the server does for `clear_text`: select-all
        # then delete.
        self.client.issue_generic_request(
            ['shell', 'input', 'keycombination', '113', '29']
        )
        time.sleep(0.5)
        self.client.press_key('KEYCODE_DEL')
        time.sleep(0.5)
      self.client.type_text(value)
      return
    converted = _to_json_action(action, screen_size)
    if converted is None:
      return
    self.client.execute_action(converted)

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    """Performs one step of the agent on the environment.

    Args:
      goal: The task goal.

    Returns:
      Done flag and per-step data for episode logging.
    """
    step_data = {
        'before_screenshot': None,
        'after_screenshot': None,
        'before_element_list': None,
        'after_element_list': None,
        'action_prompt': None,
        'action_output': None,
        'action_raw_response': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
    }
    print('----------step ' + str(len(self.history) + 1))

    task = goal.strip()
    if not task:
      # The upstream agent needs a task to open its first prompt, so fail this
      # step gracefully instead of letting the episode crash.
      print('Goal is empty; skipping the model call.')
      step_data['summary'] = 'The goal was empty, so no action was performed.'
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)
    if self._task != task:
      # `reset()` receives go_home rather than the goal, so the episode's task
      # is established lazily from the first step of each episode.
      print('Starting a new episode task; clearing the replayed history.')
      self._task = task
      self._responses = []

    state = self.get_post_transition_state()
    logical_screen_size = self.client.get_logical_screen_size()
    step_data['before_screenshot'] = state.pixels.copy()
    step_data['before_element_list'] = state.ui_elements

    encoded = base64.b64encode(infer.array_to_jpeg_bytes(state.pixels))
    image_data_url = 'data:image/jpeg;base64,' + encoded.decode('utf-8')
    messages = self._build_messages(image_data_url)
    step_data['action_prompt'] = _render_text_only(messages)

    raw_response = self._call_model(messages, self.model_args)
    step_data['action_output'] = raw_response
    step_data['action_raw_response'] = raw_response
    print('Response: ' + raw_response)
    served_action = self._normalize_payload(parse_response(raw_response))
    # The upstream agent records the turn before the harness acts on it, so the
    # model's own transcript stays consistent with what was executed.
    self._responses.append(raw_response)

    if _is_recoverable_abort(served_action):
      # Unparseable model output: keep the episode alive so the next prompt can
      # replay the failed turn and the model can correct itself.
      print('Model response could not be parsed; retrying next step.')
      step_data['summary'] = (
          'The model response was not in the expected format, so no action was'
          ' performed.'
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    # `status` actions are server-side no-ops, so completion and abort skip the
    # round trip; ANSWER is still executed because the server records the reply.
    if served_action.action not in _STATUS_ACTIONS:
      try:
        self._execute(served_action, logical_screen_size)
      except Exception as e:  # pylint: disable=broad-exception-caught
        print('Some error happened executing the action ', served_action.action)
        print(str(e))
        step_data['summary'] = (
            'Some error happened executing the action '
            + str(served_action.action)
        )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)

    if served_action.action in _EPISODE_END_ACTIONS:
      if served_action.action == ActionName.ABORT:
        print('Agent stopped since it thinks mission impossible.')
      else:
        print('Agent thinks the request has been completed.')
      step_data['summary'] = (
          'Agent thinks the request has been completed.'
          if served_action.action == ActionName.COMPLETE
          else f'Agent stopped with {served_action.action}.'
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(True, step_data)

    state = self.get_post_transition_state()
    step_data['after_screenshot'] = state.pixels.copy()
    step_data['after_element_list'] = state.ui_elements
    selected = json.dumps(served_action.payload, ensure_ascii=False)
    step_data['summary'] = f'Action selected: {selected}'
    self.history.append(step_data)

    return base_agent.AgentInteractionResult(False, step_data)


class ClientGenericR2SOL(base_agent.ClientInteractingAgent):
  """GenericAgentV2 pure-vision agent driving a Docker environment over HTTP.

  Speaks normalized 0-1000 coordinates rather than UI element indices, so it
  needs no accessibility element list and never resolves an index. The model is
  called directly so the upstream sampling parameters survive: this agent asks
  for `max_tokens: 32768`, which `infer.OpenAIWrapper` hardcodes to 1000.

  Transport follows `infer.OpenAIWrapper`: the `openai` SDK when importable,
  raw `requests` otherwise, with the same exponential backoff.
  """

  RETRY_WAITING_SECONDS = 20
  MAX_RETRY_CEILING = 5
  # Longer than OpenAIWrapper's 60s: a decision turn carries two 1080x2400 JPEG
  # data URLs and may emit up to 32768 tokens.
  MODEL_TIMEOUT_SEC = 180

  def __init__(
      self,
      client: interface.AndroidEnvClient,
      base_url: str,
      model_name: str,
      max_retry: int = 3,
      name: str = 'ClientGenericR2SOL',
  ):
    """Initializes the agent.

    Args:
      client: The Android environment client.
      base_url: Base URL of the OpenAI-compatible model endpoint. A trailing
        `/v1` is tolerated on both transports.
      model_name: Model name to request.
      max_retry: Max number of retries when a model call fails.
      name: The agent name.

    Raises:
      RuntimeError: If OPENAI_API_KEY is not set in the environment.
    """
    super().__init__(client, name)
    if 'OPENAI_API_KEY' not in os.environ:
      raise RuntimeError('OpenAI API key not set.')
    self._api_key = os.environ['OPENAI_API_KEY']
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, self.MAX_RETRY_CEILING)
    self._base_url = base_url
    self._model_name = model_name
    self.client_sdk = None
    try:
      from openai import OpenAI
      self.client_sdk = OpenAI(api_key=self._api_key, base_url=self._base_url)
    except ImportError:
      print(
          'OpenAI package not installed. Falling back to requests for model'
          ' calls.'
      )
    self.core = GenericV2Core()
    self.history = []
    self._core_task = ''
    self._step_index = 0

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    self.client.hide_automation_ui()
    self.history = []
    self.core = GenericV2Core(self.core.model_args)
    self._core_task = ''
    self._step_index = 0

  def get_post_transition_state(self) -> interface.State:
    """Gets the state, waiting for the screen to settle after an action.

    Returns:
      The state with pixels and the server-side UI elements. The accessibility
      forest stays in the container, so it is None.
    """
    if self.transition_pause is None:
      print('Waiting for screen to stabilize before grabbing state...')
      start = time.time()
      state = self.client.get_state(wait_to_stabilize=True)
      print('Fetched after %.1f seconds.', time.time() - start)
      return state
    else:
      time.sleep(self.transition_pause)
      return self.client.get_state(wait_to_stabilize=False)

  def _chat_url(self) -> str:
    """Returns the chat-completions URL, tolerating a trailing `/v1`.

    `run_on_docker.py --base_url` is conventionally given with a trailing `/v1`
    (see scripts/run_on_docker.sh), and the `openai` SDK wants it that way, but
    appending the path unconditionally would yield `/v1/v1/chat/completions`.
    """
    base = self._base_url.rstrip('/')
    if base.endswith('/v1'):
      return f'{base}/chat/completions'
    return f'{base}/v1/chat/completions'

  def _call_model(self, messages: list[dict], model_args: dict) -> str:
    """Calls the OpenAI-compatible endpoint, retrying transient failures.

    Mirrors `infer.OpenAIWrapper`: the `openai` SDK when available, a raw
    `requests` POST otherwise, both retried with exponential backoff. Unlike
    the wrapper, the caller's own `messages` and sampling parameters are passed
    through unchanged.

    Args:
      messages: Chat messages, possibly carrying image data URLs.
      model_args: Sampling parameters.

    Returns:
      The model's text response, or `infer.ERROR_CALLING_LLM` on total failure.
      That sentinel is unparseable, so the core fail-closes it into a
      recoverable ABORT and the episode takes another step. A plain string is
      returned rather than OpenAIWrapper's `(text, is_safe, raw)` triple
      because `step_data` is pickled into checkpoints and must stay
      serializable.
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self._api_key}',
    }
    payload = {'model': self._model_name, 'messages': messages, **model_args}
    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    while counter > 0:
      try:
        if self.client_sdk is None:
          response = requests.post(
              self._chat_url(),
              headers=headers,
              json=payload,
              timeout=self.MODEL_TIMEOUT_SEC,
          )
          if response.ok and 'choices' in response.json():
            return _message_text(
                response.json()['choices'][0]['message']['content']
            )
          print(
              f'Error calling model (HTTP {response.status_code}):'
              f' {response.text[:500]}'
          )
        else:
          response = self.client_sdk.chat.completions.create(
              model=self._model_name,
              messages=messages,
              timeout=self.MODEL_TIMEOUT_SEC,
              **model_args,
          )
          if response and response.choices:
            return _message_text(response.choices[0].message.content)
          print(f'Error calling model with error message: {response}')
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        print('Error calling LLM, will retry soon...')
        print(e)
      # Decremented on every failed attempt, including HTTP-level errors;
      # OpenAIWrapper only decrements on exceptions, which can spin forever on
      # a consistently failing endpoint.
      counter -= 1
      if counter > 0:
        time.sleep(wait_seconds)
        wait_seconds *= 2
    return infer.ERROR_CALLING_LLM

  def _execute(
      self, action: ActionRequest, screen_size: tuple[int, int]
  ) -> None:
    """Executes one action against the environment.

    Args:
      action: The normalized action.
      screen_size: Device logical screen size in pixels.

    Raises:
      ValueError: When the action's payload cannot be translated.
      Exception: Whatever the client raises when execution fails server-side.
    """
    if action.action == ActionName.RECENT:
      # No `json_action` equivalent exists; the recents key is the closest.
      self.client.press_key('KEYCODE_APP_SWITCH')
      return
    if action.action == ActionName.WAIT:
      # JSONAction has no duration field, so honor the requested delay here.
      # The server's own `wait` adds roughly one more second on top.
      try:
        delay = float(str(action.payload.get('value', '')).strip())
      except ValueError:
        delay = 0.0
      if delay > 0:
        time.sleep(min(delay, _MAX_WAIT_SECONDS))
      self.client.execute_action(
          json_action.JSONAction(action_type=json_action.WAIT)
      )
      return
    converted = _to_json_action(action, screen_size)
    if converted is None:
      return
    if action.action == ActionName.TYPE:
      # Focus the field, then type without submitting. `clear` mirrors what the
      # server's `input_text` does for `clear_text`: select-all then delete.
      self.client.execute_action(converted)
      time.sleep(1.0)
      if action.payload.get('clear'):
        self.client.issue_generic_request(
            ['shell', 'input', 'keycombination', '113', '29']
        )
        time.sleep(0.5)
        self.client.press_key('KEYCODE_DEL')
        time.sleep(0.5)
      self.client.type_text(_payload_value(action.payload))
      return
    self.client.execute_action(converted)

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    """Performs one step of the agent on the environment.

    Args:
      goal: The task goal.

    Returns:
      Done flag and per-step data for episode logging.
    """
    step_data = {
        'before_screenshot': None,
        'after_screenshot': None,
        'before_element_list': None,
        'after_element_list': None,
        'action_prompt': None,
        'action_output': None,
        'action_raw_response': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
    }
    print('----------step ' + str(self._step_index + 1))

    # `reset()` receives go_home rather than the goal, so the core task is
    # established lazily from the first step of each episode.
    task = goal.strip()
    if not task:
      # The core requires a non-empty task and raises otherwise, so fail this
      # step gracefully instead of letting the episode crash.
      print('Goal is empty; skipping the model call.')
      step_data['summary'] = 'The goal was empty, so no action was performed.'
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)
    if self._core_task != task:
      self.core.reset(task)
      self._core_task = task
      self._step_index = 0

    state = self.get_post_transition_state()
    logical_screen_size = self.client.get_logical_screen_size()
    step_data['before_screenshot'] = state.pixels.copy()
    step_data['before_element_list'] = state.ui_elements

    encoded = base64.b64encode(infer.array_to_jpeg_bytes(state.pixels))
    image_data_url = 'data:image/jpeg;base64,' + encoded.decode('utf-8')
    messages = self.core.build_messages(self._step_index, image_data_url)
    self.core.remember_observation_image(image_data_url)
    step_data['action_prompt'] = _render_text_only(messages)

    raw_response = self._call_model(messages, self.core.model_args)
    # `interpret` returns the action actually to be served, which may have been
    # rewritten to BACK or ABORT by the route-replan state machine.
    served_action = self.core.interpret(raw_response)
    self._step_index += 1
    step_data['action_output'] = raw_response
    step_data['action_raw_response'] = raw_response
    print('Response: ' + raw_response)

    if _is_recoverable_abort(served_action):
      # Unparseable model output: keep the episode alive so the next prompt can
      # carry the correction feedback, mirroring how T3A retries.
      print('Model response could not be parsed; retrying next step.')
      step_data['summary'] = (
          'The model response was not in the expected format, so no action was'
          ' performed.'
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    # `status` actions are server-side no-ops, so completion and abort skip the
    # round trip; ANSWER is still executed because the server records the reply.
    if served_action.action not in _STATUS_ACTIONS:
      try:
        self._execute(served_action, logical_screen_size)
      except Exception as e:  # pylint: disable=broad-exception-caught
        print('Some error happened executing the action ', served_action.action)
        print(str(e))
        step_data['summary'] = (
            'Some error happened executing the action '
            + str(served_action.action)
        )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)

    if served_action.action in _TERMINAL_ACTIONS:
      if served_action.action == ActionName.ABORT:
        print('Agent stopped since it thinks mission impossible.')
      else:
        print('Agent thinks the request has been completed.')
      step_data['summary'] = (
          'Agent thinks the request has been completed.'
          if served_action.action == ActionName.COMPLETE
          else f'Agent stopped with {served_action.action}.'
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(True, step_data)

    state = self.get_post_transition_state()
    step_data['after_screenshot'] = state.pixels.copy()
    step_data['after_element_list'] = state.ui_elements
    selected = json.dumps(served_action.payload, ensure_ascii=False)
    step_data['summary'] = f'Action selected: {selected}'
    self.history.append(step_data)

    return base_agent.AgentInteractionResult(False, step_data)
