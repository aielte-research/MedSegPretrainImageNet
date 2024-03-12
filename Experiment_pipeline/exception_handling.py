import os
import socket
import sys
import time
import traceback

LOG_DEST = '/projects/SegmentationPretraining/error_messages/' if socket.gethostname() == 'a100.cs.elte.hu' else '/data/SegmentationPretraining/error_msgs/'

def handle_exception(e : Exception, exception_message : str, log_dest : str = LOG_DEST):
        """
        Handles exception by printing out a traceback to sys.stderr, or to a file if it is too long.
        
        Parameters:
            e: Exception; the exception being raised
            exception message: str; message that should be printed when e is caught
            log_dest: str; path where the exception traceback should be logged in case it is too long to print out
        """
        print(exception_message, file = sys.stderr)
        if len(str(e)) > 1000:
            if not os.path.isdir(LOG_DEST):
                os.mkdir(LOG_DEST)

            log_file_name = log_dest + 'error_{}.txt'.format(time.time_ns())

            with open(log_file_name, 'w') as log_file:
                traceback.print_exception(type(e), e, e.__traceback__, file = log_file)
                
            ERROR_MESSAGE = 'Exception too long to print, saved to file \'{}\''.format(log_file_name)
            print(ERROR_MESSAGE, file = sys.stderr)
        else:
            traceback.print_exception(type(e), e, e.__traceback__, file = sys.stderr)
