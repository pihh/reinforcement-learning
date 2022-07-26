from tensorboardX import SummaryWriter

def create_writer(env_name,agent_name,hash):

    logdir = 'storage/environments/'+env_name+'/'+agent_name+'/'+hash
    writer = SummaryWriter(logdir=logdir+'/logs') #SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))

    return writer , logdir