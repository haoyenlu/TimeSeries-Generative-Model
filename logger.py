class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



class Logger:
    def __init__(self):
        pass

    
    def warning(self,text):
        print(bcolors.WARNING + "[Warning]:" + f"[{text}]" + bcolors.ENDC)
        

    def info(self,text):
        print(bcolors.OKBLUE + "[INFO]:" + f"[{text}]" + bcolors.ENDC)

    def debug(self,text):
        print(bcolors.OKGREEN + "[DEBUG]:" + f"[{text}]" + bcolors.ENDC)