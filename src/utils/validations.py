#from dotwiz import DotWiz

def validate_action_space(env, valid_spaces=["discrete",'continuous']):
    if hasattr(env.action_space, 'n'):
        action_space_type = 'discrete'
    else:
        action_space_type = 'continuous'

    if not action_space_type in valid_spaces:
        raise Exception('This algorithm doesn\'t support {} action spaces'.format(action_space_type))

