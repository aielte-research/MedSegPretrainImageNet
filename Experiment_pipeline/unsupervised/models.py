import utils
from model import Model

import torch

class MeanTeacherModel(torch.nn.Module):
    
    @staticmethod
    def fill_kwargs(config_dict):
        utils.fill_dict(config_dict['student_model'])
    
    def __init__(self, student_model, *args, **kwargs):
        super().__init__()
        
        self.student = utils.create_object_from_dict(student_model, wrapper_class = Model, *args, **kwargs)
        self.teacher = utils.create_object_from_dict(student_model, wrapper_class = Model, *args, **kwargs)
        
        self.teacher.requires_grad_(False)
        self.PASS_ALL_INPUT = getattr(self.student, 'PASS_ALL_INPUT', False)
    
    def init_weights(self, *args, **kwargs):
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            teacher_param.data.copy_(student_param.data)
    
    def forward(self, *args, **kwargs):
        return self.student(*args, **kwargs), self.teacher(*args, **kwargs)
    
    def parameters(self, recurse = True):
        return self.student.parameters(recurse), lambda: zip(self.student.parameters(), self.teacher.parameters())
    
    def get_num_params(self):
        return 2 * sum(dict((p.data_ptr(), p.numel()) for p in self.student.parameters()).values())