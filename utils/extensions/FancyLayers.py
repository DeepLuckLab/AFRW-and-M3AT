from torch.nn import Module

def __repr__(self):
    def _addindent(s_, numSpaces):

        s = s_.split('\n')

        if len(s) == 1:
            return s_

        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s

    def is_trainable(module):
        try:
            params = list(module.parameters())

            if not params:
                return None
        except:

            return None

        requires_grads = [p.requires_grad for p in params]
        if all(requires_grads):
            return True
        elif not any(requires_grads):
            return False
        return None

    extra_lines = []
    extra_repr = self.extra_repr()

    if extra_repr:
        extra_lines = extra_repr.split('\n')

    child_lines = []
    for key, module in self._modules.items():

        trainable = is_trainable(module)

        prefix = ''
        if trainable == True:
            prefix = '🔥 '
        elif trainable == False:
            prefix = '⛄️ '
        else:
            prefix = '   '

        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        child_lines.append(prefix + '(' + key + '): ' + mod_str)

    lines = extra_lines + child_lines

    main_str = self._get_name() + '('

    if lines:

        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]

        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'

    return main_str

Module.__repr__ = __repr__