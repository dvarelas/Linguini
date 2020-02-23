import sys
import datetime
import os


from linguini import BASE_PATH


class TrainingLogger(object):
    """
    Training logger
    """
    def __init__(self, exp_name):
        self._mkdir()
        self.exp_name = exp_name
        self.terminal = sys.stdout
        self.log = open(self.exp_name, "a")
        self.log.write(
            "date and time = " + str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

    def _mkdir(self):
        """
        Creates directory for logs if it does not exist
        :return:
        """
        if not os.path.exists(BASE_PATH + 'logs/'):
            os.makedirs(BASE_PATH + 'logs/')

    def write(self, message):
        """
        Writes the logs a text file
        :param message:
        :return:
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
