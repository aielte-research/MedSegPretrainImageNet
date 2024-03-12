import re

# objects here implement changes in the coefficients of combined losses


def convert_to_snake(match):
    return '_' + match.group(0).lower()

def get_idcs(loss_names_to_change, all_loss_names):
    if isinstance(loss_names_to_change, str):
        loss_names_to_change = [loss_names_to_change]
    loss_names_to_change = [
        re.sub('[A-Z]', convert_to_snake, loss).strip('_').replace('__', '_').replace(' ', '_')
        for loss in loss_names_to_change
        ]
    idcs = [all_loss_names.index(loss) for loss in loss_names_to_change]
    if len(idcs) == 1:
        idcs = idcs[0]
    return idcs
    

class SingularChange(object):
    """
    Callable object that changes the coefficient of one loss every few epochs.

    Parameters:
        loss_to_change: str; name of the loss function whose coefficient should be changed
        change_fn: callable (float --> float) object that takes the current coefficient and increases it
        step_frequency: int or None; if not None, the coefficients will only change every step_frequency epochs
    """

    def __init__(self, loss_to_change, change_fn, step_frequency = 1, loss_names = [], *args, **kwargs):
        self.loss_idx = get_idcs(loss_to_change, loss_names)
        self.change_fn = change_fn
        self.step_frequency = step_frequency
    
    def __call__(self, epoch_idx, curr_coeffs):
        if self.step_frequency and epoch_idx % self.step_frequency == 0:
            return [weight if i != self.loss_idx else self.change_fn(weight) for i, weight in enumerate(curr_coeffs)]
        return curr_coeffs

class LinearChange(SingularChange):
    """
    Callable object that increases or decreases the coefficient of one loss linearly.

    Parameters:
        loss_to_change: str; name of the loss function whose coefficient should be changed
        change_factor: float; amount that will be added to the coefficient every time
        step_frequency: int or None; if not None, the coefficients will only change every step_frequency epochs
    """

    PARAMS = {
        'loss to change': {
            'argument name': 'loss_to_change',
            'default': 'boundary_loss'
            },
        'change factor': {
            'argument name': 'change_factor',
            'default': 0.1
            },
        'step frequency': {
            'argument name': 'step_frequency',
            'default': 1
            }
        }

    def __init__(self, config_dict, *args, **kwargs):
        loss_to_change = config_dict['loss to change']
        change_factor = config_dict['change factor']
        step_frequency = config_dict['step frequency']
        super().__init__(loss_to_change, change_fn = lambda x: x + change_factor,
                         step_frequency = step_frequency, *args, **kwargs)
    
class LinearRebalance(object):
    """
    Callable object that increases the coefficient of one loss and decreases another, linearly, so the sum of their coefficients does not change.

    Parameters:
        loss_to_increase: str; name of the loss whose coefficient should be increased
        loss_to_decrease: str; name of the loss whose coefficient should be decreased
        increase_factor: float; amount that will be added to the coefficient of loss_to_increase and substracted from the coefficient of loss_to_decrease
        step_frequency: int or None; if not None, the coefficients will only change every step_frequency epochs
    """

    PARAMS = {
        'loss to increase': 'boundary_loss',
        'loss to decrease': 'tversky_loss',
        'increase factor': 0.1,
        'step frequency': 1
    }

    def __init__(self, config_dict, loss_names = [], *args, **kwargs):
        loss_to_decrease = config_dict['loss to increase']
        loss_to_increase = config_dict['loss to decrease']
        self.increase_idx, self.decrease_idx = get_idcs((loss_to_increase, loss_to_decrease), loss_names)
        self.increase_factor = config_dict['increase factor']
        self.step_frequency = config_dict['step frequency']
    
    def __call__(self, epoch_idx, curr_coeffs):
        if self.step_frequency and epoch_idx % self.step_frequency == 0:
            weights = []
            for i, weight in enumerate(curr_coeffs):
                if i == self.increase_idx:
                    weights.append(weight + self.increase_factor)
                elif i == self.decrease_idx:
                    weights.append(weight - self.increase_factor)
                else:
                    weights.append(weight)
            return weights
        return curr_coeffs

class DiscontinousChange(object):
    """
    Callable object that changes the coefficients of different losses in a combined loss.

    Parameters:
        change_dict: dict(int, dict(str, float)); dictionary that has keys corresponding to the epoch indices (startig from one) when the coefficients should change, and values that are dictionaries containing an updated coefficient dictionary
    """

    def __init__(self, config_dict, loss_names = []):
        self.change_dict = config_dict.value()
        self.loss_names = loss_names
    
    def __call__(self, epoch_idx, curr_coeffs):
        if epoch_idx in self.change_dict:
            curr_change_dict = self.change_dict[epoch_idx]
            losses_to_change = list(curr_change_dict.keys())
            change_idcs = get_idcs(losses_to_change, self.loss_names)
            weights = []
            for i, weight in enumerate(curr_coeffs):
                if i in change_idcs:
                    weights.append(curr_change_dict[losses_to_change[i]])
                else:
                    weights.append(weight)
                return weights
        return curr_coeffs