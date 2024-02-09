
def message(*args, verbose=True):
    """ Print date - message """
    def get_time():
        """ Get time in the following format:  25/06/2021 07:58:56 """
        from datetime import datetime
        now = datetime.now()
        return(now.strftime("%d/%m/%Y %H:%M:%S"))
    if verbose:
        date = get_time()
        args = [str(i) for i in args]
        #print(date, ' - ', ''.join(args))
        print(' - ', ' '.join(args))