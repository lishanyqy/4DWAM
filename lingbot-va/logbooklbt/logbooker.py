from logbook import Logger,FileHandler,NOTSET
from logbook.more import ColorizedStderrHandler
from pathlib import Path
# import const.constants as const
from logbook._termcolors import colorize
import os
import sys
from logbook import(
    CRITICAL,
    DEBUG,
    ERROR,
    INFO,
    NOTICE,
    NOTSET,
    TRACE,
    WARNING,
)

route = 'fedlog/logfiles/client.log'
LOG_FILE_FORMAT = '({record.level_name})[{record.time:%Y-%m-%d %H:%M:%S}] {record.filename}:{record.lineno} {record.module}: {record.message}'

class ColoredConsoleHandler(ColorizedStderrHandler):
    def get_color(self, record):
        """Returns the color for this record."""
        if record.level >= ERROR:
            return "red"
        elif record.level >= NOTICE:
            return "yellow"
        elif record.level >= INFO:
            return 'green'
        return "lightgray"
    def format(self, record):
        rv = super().format(record)
        if self.should_colorize(record):
            color = self.get_color(record)
            if color:
                rv = colorize(color, rv)
        return rv

class Attender(Logger):
    def __init__(self,index='client',filePath = 'file.log',level = NOTSET,handlers=None) -> None:
        super().__init__(index,level)
        if not os.path.exists(filePath):
            fp = Path(filePath)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.touch()
        self.handlers=[FileHandler(
            filename = filePath,
            format_string = LOG_FILE_FORMAT,
            bubble=True,
        )]
        if handlers is not None:
            if isinstance(handlers,list):
                self.handlers.extend(handlers)
            else:
                self.handlers.append(handlers)
                
    def get_color(self, record):
        """Returns the color for this record."""
        if record.level >= ERROR:
            return "red"
        elif record.level >= NOTICE:
            return "yellow"
        elif record.level >= INFO:
            return 'green'
        return "lightgray"
    
    def format(self, record):
        rv = super().format(record)
        if self.should_colorize(record):
            color = self.get_color(record)
            if color:
                rv = colorize(color, rv)
        return rv
    
    
    
glogger = Attender(index='global',filePath='global.log',handlers=ColoredConsoleHandler(bubble=True,format_string=LOG_FILE_FORMAT))

if __name__ == '__main__':
    # chandler = ColorizedStderrHandler()
    # fhandler = TimedRotatingFileHandler(
    #     filename='fedlog/logfiles/client.log'
    # )
    # clogger = Attender(handlers=[chandler,fhandler])
    glogger.info('test')
    glogger.warn('warn')
    # clogger.info('this is a info log')
    # clogger.warn('this is a warn log')