import os
import numbers
from tensorboardX import SummaryWriter
#from tensorflow import summary

from datetime import datetime

def mkdir(dir):
    exists = os.path.exists(dir)

    if not exists:
        # Create a new directory because it does not exist 
        os.makedirs(dir)

def create_writer(env_name,hash):

    configdir = 'storage/environments/'+env_name+'/'+hash
    logdir = configdir+'/logs/'+datetime.now().strftime("%Y%m%d-%H%M%S")
    
    mkdir(configdir)
    mkdir(logdir)

    writer = SummaryWriter(logdir=logdir) #SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))
    #writer = summary.create_file_writer(logdir)
    return writer , configdir

def scalar(instance, scalar_name,score, steps):
    if hasattr(instance,'tensorboard_writer') and isinstance(score,numbers.Number) and isinstance(steps,numbers.Number):
        instance.tensorboard_writer.add_scalar('Data/'+scalar_name,score, steps)
        #with instance.tensorboard_writer.as_default():
            #summary.scalar('Data/'+scalar_name,score, steps)

def histogram(instance, name, model):
    print('@TODO tensorboard histogram')
    # if hasattr(instance,'tensorboard_writer'):
    #     with instance.tensorboard_writer.as_default():
    #         for layer in model.layers:
    #             instance.tensorboard_writer.add_histogram('Data/'+model.name+'/'+layer.name,layer.get_weights())
    #             # summary.histogram(
    #             #     model.name+'/'+layer.name,
    #             #     layer.get_weights()
    #             #     # ''+name+'/'+model.name'/'+ layer.name, 
    #             #     # layer.get_weights() 
    #             # )
                

def graph(instance):
    print('@TODO tensorboard graph')
    # if hasattr(instance,'tensorboard_writer') and hasattr(instance,'models'):
    #     all_valid = True
    #     invalid_models = []
    #     for model in instance.models:
    #         try:
    #             with instance.tensorboard_writer.as_default():
    #                 summary.graph(model)
    #         except Exception as e:
    #             #print(e)
    #             all_valid = False 
    #             invalid_models.append(model.name)
    #             #print(model.name +' is not compatible with tensorboard graph')
        
    #     if not all_valid:
    #         print('The following models are incompatible with tensorboard graphs', invalid_models)
    #         print()