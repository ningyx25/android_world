from absl import app
from absl import flags

from android_world import checkpointer as checkpointer_lib
from android_world import constants
from android_world import suite_utils

_METADATA_FIELDS = [
    constants.EpisodeConstants.GOAL,
    constants.EpisodeConstants.TASK_TEMPLATE,
    constants.EpisodeConstants.INSTANCE_ID,
    constants.EpisodeConstants.IS_SUCCESSFUL,
    constants.EpisodeConstants.EPISODE_LENGTH,
    constants.EpisodeConstants.RUN_TIME,
    constants.EpisodeConstants.EXCEPTION_INFO,
]
   
_CHECKPOINT_DIR = flags.DEFINE_string(
  'checkpoint_dir',
  '',
  'The directory to save checkpoints and resume evaluation from. If the'
  ' directory contains existing checkpoint files, evaluation will resume from'
  ' the latest checkpoint. If the directory is empty or does not exist, a new'
  ' directory will be created.',
)

def print_scoreboard():
    checkpoint_dir = _CHECKPOINT_DIR.value
    checkpointer = checkpointer_lib.IncrementalCheckpointer(checkpoint_dir)
    print(f'Loading checkpoint from {checkpoint_dir}')
    prior_episodes = checkpointer.load(fields=_METADATA_FIELDS)
    if prior_episodes:
        print(f'\n=== Checkpoint summary ({len(prior_episodes)} episodes) ===')
        suite_utils.process_episodes(prior_episodes, print_summary=True)
    else:
        print('No episodes found in checkpoint.')

def main(argv: list[str]) -> None:
    del argv
    print_scoreboard()


if __name__ == '__main__':
  app.run(main)
