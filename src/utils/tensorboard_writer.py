import numbers
from tensorboardX import SummaryWriter
from tensorflow import summary

from datetime import datetime
def create_writer(env_name,agent_name,hash):

    logdir = 'storage/environments/'+env_name+'/logs/'+hash+'__'+datetime.now().strftime("%Y%m%d-%H%M%S")
    #writer = SummaryWriter(logdir=logdir+'/logs') #SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))
    writer = summary.create_file_writer(logdir+'/logs')
    return writer , logdir

def scalar(instance, scalar_name,score, steps):
    if hasattr(instance,'tensorboard_writer') and isinstance(score,numbers.Number) and isinstance(steps,numbers.Number):
        with instance.tensorboard_writer.as_default():
            summary.scalar('Data/'+scalar_name,score, steps)

def histogram(instance, name, model):
    if hasattr(instance,'tensorboard_writer'):
        with instance.tensorboard_writer.as_default():
            for layer in model.layers:
                summary.histogram(
                    model.name+'/'+layer.name,
                    layer.get_weights()
                    # ''+name+'/'+model.name'/'+ layer.name, 
                    # layer.get_weights() 
                )
                

def graph(instance):
    if hasattr(instance,'tensorboard_writer') and hasattr(instance,'models'):
        all_valid = True
        invalid_models = []
        for model in instance.models:
            try:
                with instance.tensorboard_writer.as_default():
                    summary.graph(model)
            except Exception as e:
                #print(e)
                all_valid = False 
                invalid_models.append(model.name)
                #print(model.name +' is not compatible with tensorboard graph')
        
        if not all_valid:
            print('The following models are incompatible with tensorboard graphs', invalid_models)
            print()