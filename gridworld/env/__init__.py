from .darkroom import sample_darkroom, sample_darkroom_permuted, Darkroom, DarkroomPermuted, map_dark_states, map_dark_states_inverse
from .dark_key_to_door import DarkKeyToDoor, sample_dark_key_to_door


ENVIRONMENT = {
    'darkroom': Darkroom,
    'darkroompermuted': DarkroomPermuted,
    'darkkeytodoor': DarkKeyToDoor,
}


SAMPLE_ENVIRONMENT = {
    'darkroom': sample_darkroom,
    'darkroompermuted': sample_darkroom_permuted,
    'darkkeytodoor': sample_dark_key_to_door,
}


def make_env(config, **kwargs):
    def _init():
            return ENVIRONMENT[config['env']](config, **kwargs)
    return _init