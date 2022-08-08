import os
import json
import numbers

from pathlib import Path
from datetime import datetime

from tensorboardX import SummaryWriter

class ResultsWriter:
    def __init__(self,env_name,agent_hash):
        self.env_name = env_name
        self.agent_hash = agent_hash
        self.file_path = 'storage/environments/'+env_name+'results.json'
        self.score = False

        # schedule.every(10).seconds.do(self.log)
        # while True:
        #     schedule.run_pending()
        #     time.sleep(1)

    def load_best_score(self):
        file_path = 'storage/environments/'+self.env_name+'/results.json'
        score = False
        try:
            with open(file_path, 'r') as fp:
                results = json.load(fp)
                score = results[self.agent_hash]
        except IOError:
            # print('File not found, will create a new one.')
            pass

        return score

    # def queue(self,score):
    #     self.score = score
    #     self.log()
        # if self.score == False:
        #     self.score = score
        # else:
        #     if self.score < score:
        #         self.score = score 

    def log(self,score):
        #if self.score != False:
            #print('log',self.check_file())
            #if not self.check_file():
            #    log_environment_results_file(self.env_name,self.agent_hash,self.score)
            #    self.score = False
            #score = self.score            
        log_environment_results_file(self.env_name,self.agent_hash,score)
        #self.score = False

    # def check_file(self):
    #     file = Path(self.file_path)
    #     if file.exists() and file.stat().st_size >= 75:
    #         return True
    #     else:
    #         return False
    def store_test_results(self,agent,results_dataframe):
        results_dataframe.to_csv(agent.writer_log_directory+'/results__'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv')


def log_environment_results_file(env_name,agent_hash,score):
    file_path = 'storage/environments/'+env_name+'/results.json'
    try:
        with open(file_path, 'r') as fp:
            data = json.load(fp)

    except IOError:
        # print('File not found, will create a new one.')
        data = {}

    # do stuff with your data...
    with open(file_path, 'w') as fp:
        if agent_hash in data:
            if data[agent_hash] < score:
                data[agent_hash] = score
        else:
            data[agent_hash] = score
            
        json.dump(data, fp, indent=4)

def mkdir(dir):
    exists = os.path.exists(dir)

    if not exists:
        # Create a new directory because it does not exist 
        os.makedirs(dir)

def create_writer(env_name,hash):

    configdir = 'storage/environments/'+env_name+'/'+hash
    modelsdir = configdir + '/models'
    logdir = configdir+'/logs/'+datetime.now().strftime("%Y%m%d-%H%M%S")
    
    mkdir(configdir)
    mkdir(modelsdir)
    mkdir(logdir)

    writer = SummaryWriter(logdir=logdir) #SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))
    #writer = summary.create_file_writer(logdir)
    return writer , configdir

def scalar(instance, scalar_name,score, steps):
    if hasattr(instance,'writer') and isinstance(score,numbers.Number) and isinstance(steps,numbers.Number):
        instance.writer.add_scalar('Data/'+scalar_name,score, steps)
        #with instance.writer.as_default():
            #summary.scalar('Data/'+scalar_name,score, steps)

def histogram(instance, name, model):
    print('@TODO tensorboard histogram')
    # if hasattr(instance,'writer'):
    #     with instance.writer.as_default():
    #         for layer in model.layers:
    #             instance.writer.add_histogram('Data/'+model.name+'/'+layer.name,layer.get_weights())
    #             # summary.histogram(
    #             #     model.name+'/'+layer.name,
    #             #     layer.get_weights()
    #             #     # ''+name+'/'+model.name'/'+ layer.name, 
    #             #     # layer.get_weights() 
    #             # )
                

def graph(instance):
    print('@TODO tensorboard graph')
    # if hasattr(instance,'writer') and hasattr(instance,'models'):
    #     all_valid = True
    #     invalid_models = []
    #     for model in instance.models:
    #         try:
    #             with instance.writer.as_default():
    #                 summary.graph(model)
    #         except Exception as e:
    #             #print(e)
    #             all_valid = False 
    #             invalid_models.append(model.name)
    #             #print(model.name +' is not compatible with tensorboard graph')
        
    #     if not all_valid:
    #         print('The following models are incompatible with tensorboard graphs', invalid_models)
    #         print()