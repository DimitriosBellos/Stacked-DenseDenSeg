class Colors(object):
    def __init__(self):
        self.none = '\033[0m'
        self.black = '\033[0;30m'
        self.red = '\033[0;31m'
        self.green = '\033[0;32m'
        self.yellow = '\033[0;33m'
        self.blue = '\033[0;34m'
        self.magenta = '\033[0;35m'
        self.cyan = '\033[0;36m'
        self.white = '\033[0;37m'
        self.Black = '\033[1;30m'
        self.Red = '\033[1;31m'
        self.Green = '\033[1;32m'
        self.Yellow = '\033[1;33m'
        self.Blue = '\033[1;34m'
        self.Magenta = '\033[1;35m'
        self.Cyan = '\033[1;36m'
        self.White = '\033[1;37m'
        self._black = '\033[40m'
        self._red = '\033[41m'
        self._green = '\033[42m'
        self._yellow = '\033[43m'
        self._blue = '\033[44m'
        self._magenta = '\033[45m'
        self._cyan = '\033[46m'
        self._white = '\033[47m'
        super(Colors, self).__init__()

    def __call__(self):
        return self
