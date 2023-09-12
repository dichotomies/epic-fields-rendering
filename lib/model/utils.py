
def set_requires_grad(model, keys_incl=None, keys_excl=None, requires_grad=True):
    assert (keys_incl is not None and keys_excl is not None) == False
    for p in model.named_parameters():
        assert type(p[0]) is str
        if keys_incl is None and keys_excl is None:
            p[1].requires_grad = requires_grad
        elif keys_incl is not None:
            if any([k in p[0] for k in keys_incl]):
                p[1].requires_grad = requires_grad
        elif keys_excl is not None:
            if any([k in p[0] for k in keys_excl]):
                print(f'Parameter {p[0]} is not set to requires_grad={requires_grad}.')
                pass
            else:
                p[1].requires_grad = requires_grad