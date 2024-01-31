from .base import VerbalizedEnv


def load_verbenv_by_name(name) -> VerbalizedEnv:
    if name == 'touchdown':
        from .touchdown import VerbalizedTouchdown as VEnv
    elif name == 'descriptive_touchdown':
        from .touchdown import DescriptiveVerbalizedTouchdown as VEnv
    elif name == 'alfworld':
        from .alfworld import VerbalizedALFWorld as VEnv
    elif name == 'scienceworld':
        from .scienceworld import VerbalizedScienceWorld as VEnv
    elif name == 'sparse_scienceworld':
        from .scienceworld import SparseVerbalizedScienceWorld as VEnv
    else:
        raise NotImplementedError('Cannot load {}'.format(name))
    return VEnv
