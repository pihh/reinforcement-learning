class Buffer:
    def __init__(self,keys):
        self.keys = keys
        self.__init_memory()

    def __init_memory(self):
        self.memory = {}
        for key in self.keys:
            self.memory[key] = []

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()


    def store(self,key,value):
        self.memory[key].append(value)

    def get(self,key):
        return self.memory[key]